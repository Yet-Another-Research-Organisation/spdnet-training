"""Enhanced metrics utilities for SPDNet training with energy and resource monitoring."""

import json
import time
from pathlib import Path
from typing import Dict, Any
import psutil
import torch
import threading
import numpy as np

try:
    from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain, RaplDevice
    from pyJoules.device.nvidia_device import NvidiaGPUDomain, NvidiaGPUDevice
    from pyJoules.energy_meter import EnergyMeter
    PYJOULE_AVAILABLE = True
except ImportError:
    PYJOULE_AVAILABLE = False


class EnhancedMetricsWriter:
    """
    Enhanced metrics writer with energy monitoring, covariance analysis,
    and comprehensive resource tracking.
    """

    def __init__(
        self,
        log_dir: str = "training_logs",
        experiment_name: str = "experiment",
        monitoring_interval: float = 1.0,
    ):
        """
        Initialize enhanced metrics writer.

        Args:
            log_dir: Directory to store metrics files
            experiment_name: Name of the experiment
            monitoring_interval: Interval for energy/RAM sampling (seconds)
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create metrics files
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.jsonl"
        self.status_file = self.log_dir / f"{experiment_name}_status.json"
        self.covariance_stats_file = self.log_dir / f"{experiment_name}_covariance_stats.jsonl"

        # Track process
        self.process = psutil.Process()
        self.start_ram_usage = self.process.memory_info().rss / (1024**3)  # GB
        self.start_time = time.time()

        # Energy and RAM monitoring
        self.energy_monitoring_active = False
        self.monitoring_thread = None
        self.energy_samples = []
        self.ram_samples = []
        self.monitoring_interval = monitoring_interval

        # pyJoules setup
        self.pyjoule_available = PYJOULE_AVAILABLE
        self.energy_domains = []
        self.energy_devices = []

        if PYJOULE_AVAILABLE:
            self._setup_pyjoules()
        else:
            print("⚠ pyJoules not available. Energy monitoring disabled.")

        # Initialize status
        self._write_status({
            "experiment_name": experiment_name,
            "status": "initialized",
            "start_time": self.start_time,
            "current_epoch": 0,
            "total_epochs": 0,
            "best_val_acc": 0.0,
            "best_val_loss": float("inf"),
        })

    def _setup_pyjoules(self):
        """Setup pyJoules energy monitoring devices."""
        try:
            # Setup RAPL domains
            rapl_domains = []

            # Try package domain (CPU)
            try:
                domain = RaplPackageDomain(0)
                test_device = RaplDevice()
                test_device.configure(domains=[domain])
                rapl_domains.append(domain)
            except Exception:
                pass

            # Try DRAM domain
            try:
                domain = RaplDramDomain(0)
                test_device = RaplDevice()
                test_device.configure(domains=[domain])
                rapl_domains.append(domain)
            except Exception:
                pass

            # Create RAPL device
            if rapl_domains:
                rapl_device = RaplDevice()
                rapl_device.configure(domains=rapl_domains)
                self.energy_devices.append(rapl_device)
                self.energy_domains.extend(rapl_domains)

            # Try to add GPU
            if torch.cuda.is_available():
                try:
                    gpu_domain = NvidiaGPUDomain(0)
                    gpu_device = NvidiaGPUDevice()
                    gpu_device.configure(domains=[gpu_domain])
                    self.energy_devices.append(gpu_device)
                    self.energy_domains.append(gpu_domain)
                    print("✓ GPU energy monitoring enabled")
                except Exception as e:
                    print(f"ℹ GPU energy monitoring not available: {e}")

            print(f"✓ pyJoules initialized: {len(self.energy_devices)} device(s), {len(self.energy_domains)} domain(s)")
        except Exception as e:
            print(f"⚠ Could not initialize pyJoules: {e}")
            self.pyjoule_available = False

    def _monitoring_loop(self):
        """Background thread for continuous energy and RAM monitoring."""
        if not self.pyjoule_available or not self.energy_devices:
            return

        try:
            meter = EnergyMeter(self.energy_devices)

            while self.energy_monitoring_active:
                try:
                    meter.start()
                    time.sleep(self.monitoring_interval)
                    meter.stop()
                except (PermissionError, OSError) as e:
                    if 'Permission denied' in str(e) or e.errno == 13:
                        print("\n⚠ Cannot access RAPL energy counters (Permission denied).")
                        print("Energy monitoring disabled. Configure permissions:")
                        print("  sudo chmod -R a+r /sys/class/powercap/intel-rapl")
                        self.energy_monitoring_active = False
                        self.pyjoule_available = False
                        break
                    else:
                        print(f"⚠ Error reading energy: {e}")
                        time.sleep(self.monitoring_interval)
                        continue
                except Exception as e:
                    print(f"⚠ Error reading energy: {e}")
                    time.sleep(self.monitoring_interval)
                    continue

                end_time = time.time()

                # Get trace
                trace = meter.get_trace()
                if trace and len(trace) > 0:
                    sample = trace[-1]
                    energy_data = {
                        'timestamp': end_time,
                        'duration': sample.duration / 1e6,  # µs to s
                    }

                    # Extract energy values
                    for tag, energy_raw in sample.energy.items():
                        device_name = str(tag).lower()
                        for suffix in ['_0', 'domain', 'rapl']:
                            device_name = device_name.replace(suffix, '')
                        device_name = device_name.strip('_')

                        # Convert to Joules
                        if 'package' in device_name:
                            device_name = 'cpu'
                            energy_joules = energy_raw / 1e6  # µJ to J
                        elif 'nvidia' in device_name or device_name == 'gpu':
                            device_name = 'gpu'
                            energy_joules = energy_raw / 1e3  # mJ to J
                        elif 'dram' in device_name:
                            device_name = 'ram'
                            energy_joules = energy_raw / 1e6  # µJ to J
                        else:
                            energy_joules = energy_raw / 1e6

                        energy_data[f'{device_name}_joules'] = energy_joules

                    self.energy_samples.append(energy_data)

                # Record RAM usage
                ram_gb = self.process.memory_info().rss / (1024**3)
                self.ram_samples.append({
                    'timestamp': end_time,
                    'ram_gb': ram_gb
                })

        except Exception as e:
            print(f"⚠ Error in energy monitoring loop: {e}")
            self.energy_monitoring_active = False

    def start_energy_monitoring(self):
        """Start continuous energy and RAM monitoring in background thread."""
        if not self.pyjoule_available:
            return

        if not self.energy_domains:
            print("⚠ No energy domains configured")
            return

        if self.energy_monitoring_active:
            return

        self.energy_monitoring_active = True
        self.energy_samples = []
        self.ram_samples = []

        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        print(f"✓ Energy monitoring started (sampling every {self.monitoring_interval}s)")

    def stop_energy_monitoring(self) -> Dict[str, Any]:
        """Stop energy monitoring and compute statistics."""
        if not self.energy_monitoring_active:
            return {}

        self.energy_monitoring_active = False

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        stats = {
            'samples_count': len(self.energy_samples),
            'monitoring_duration_seconds': 0,
        }

        if self.energy_samples:
            stats['monitoring_duration_seconds'] = (
                self.energy_samples[-1]['timestamp'] - self.energy_samples[0]['timestamp']
            )

            # Aggregate energy by component
            components = set()
            for sample in self.energy_samples:
                for key in sample.keys():
                    if key.endswith('_joules'):
                        component = key.replace('_joules', '')
                        components.add(component)

            for component in components:
                energies = [s[f'{component}_joules'] for s in self.energy_samples if f'{component}_joules' in s]
                if energies:
                    stats[f'{component}_total_joules'] = sum(energies)
                    stats[f'{component}_mean_watts'] = np.mean(energies) / self.monitoring_interval
                    stats[f'{component}_max_watts'] = max(energies) / self.monitoring_interval

        if self.ram_samples:
            ram_values = [s['ram_gb'] for s in self.ram_samples]
            stats['ram_mean_gb'] = np.mean(ram_values)
            stats['ram_max_gb'] = max(ram_values)
            stats['ram_min_gb'] = min(ram_values)

        return stats

    def log_metrics(self, epoch: int, step: int, metrics: Dict[str, Any]):
        """Log training metrics."""
        entry = {
            "timestamp": time.time(),
            "epoch": epoch,
            "step": step,
            **metrics
        }

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_covariance_stats(self, epoch: int, stats: Dict[str, Any]):
        """Log covariance matrix statistics."""
        entry = {
            "timestamp": time.time(),
            "epoch": epoch,
            **stats
        }

        with open(self.covariance_stats_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_model_info(self, **kwargs):
        """Log model information."""
        self.update_status({"model_info": kwargs})

    def update_status(self, status_update: Dict[str, Any]):
        """Update training status."""
        if self.status_file.exists():
            with open(self.status_file, "r") as f:
                current_status = json.load(f)
        else:
            current_status = {}

        current_status.update(status_update)
        current_status["last_update"] = time.time()

        self._write_status(current_status)

    def _write_status(self, status: Dict[str, Any]):
        """Write status to file."""
        with open(self.status_file, "w") as f:
            json.dump(status, f, indent=2)

    def get_training_duration(self) -> float:
        """Get total training duration in seconds."""
        return time.time() - self.start_time

    def get_ram_usage(self) -> float:
        """Get current RAM usage in GB."""
        return self.process.memory_info().rss / (1024**3)


def compute_covariance_stats(cov_matrices: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics for covariance matrices.

    Args:
        cov_matrices: Batch of covariance matrices [B, C, C]

    Returns:
        Dictionary with eigenvalue and conditioning statistics
    """
    if cov_matrices.dim() != 3:
        raise ValueError(f"Expected 3D tensor [B, C, C], got shape {cov_matrices.shape}")

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(cov_matrices)  # [B, C]

    # Compute statistics
    stats = {
        'min_eigenvalue': eigenvalues.min().item(),
        'max_eigenvalue': eigenvalues.max().item(),
        'mean_eigenvalue': eigenvalues.mean().item(),
        'std_eigenvalue': eigenvalues.std().item(),
    }

    # Condition number (ratio of max to min eigenvalue)
    condition_numbers = eigenvalues.max(dim=1)[0] / (eigenvalues.min(dim=1)[0] + 1e-10)
    stats['mean_condition_number'] = condition_numbers.mean().item()
    stats['max_condition_number'] = condition_numbers.max().item()
    stats['std_condition_number'] = condition_numbers.std().item()

    # Determinant
    determinants = torch.linalg.det(cov_matrices)
    stats['mean_determinant'] = determinants.mean().item()
    stats['min_determinant'] = determinants.min().item()

    # Trace
    traces = torch.diagonal(cov_matrices, dim1=1, dim2=2).sum(dim=1)
    stats['mean_trace'] = traces.mean().item()

    # Frobenius norm
    frobenius_norms = torch.norm(cov_matrices, p='fro', dim=(1, 2))
    stats['mean_frobenius_norm'] = frobenius_norms.mean().item()

    return stats
