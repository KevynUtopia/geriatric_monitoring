from typing import Optional
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrices(summary_results, save_dir):
    """Placeholder for confusion matrix plots."""
    pass

def plot_recall_by_identity(identity_results, save_dir):
    """Placeholder for recall by identity plots."""
    pass

def plot_f2_score_analysis(identity_results, summary_results, save_dir):
    """Placeholder for F2 score analysis plots."""
    pass

def plot_human_vs_system_timeseries(
    recording_session: str,
    timestamps_system: np.ndarray,
    predictions_system: np.ndarray,
    timestamps_human: np.ndarray,
    labels_human: np.ndarray,
    save_dir: str,
    figsize: tuple = (14, 4)
) -> Optional[str]:
    """
    Plot two curves across time: human aggregated anomaly events (0/1) and system anomaly score.
    Saves figure to save_dir and returns the file path.
    """
    try:
        if timestamps_system.size == 0 or predictions_system.size == 0:
            return None
        # Sort both series by time
        sys_order = np.argsort(timestamps_system)
        ts_sys = timestamps_system[sys_order]
        ys_sys = predictions_system[sys_order]

        # Human series
        if timestamps_human.size > 0 and labels_human.size > 0:
            human_order = np.argsort(timestamps_human)
            ts_h = timestamps_human[human_order]
            ys_h = labels_human[human_order]
        else:
            ts_h = np.array([])
            ys_h = np.array([])

        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(ts_sys, ys_sys, label='GCL anomaly score (system)', color='C0', linewidth=1.5)
        if ts_h.size > 0:
            ax.plot(ts_h, ys_h, label='Aggregated human anomaly (0/1)', color='C3', linewidth=1.2, alpha=0.9)

        ax.set_title(f"Human vs System Anomaly Time-Series\n{recording_session}")
        ax.set_xlabel('Timestamp (seconds)')
        ax.set_ylabel('Score / Label')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right')
        fig.tight_layout()

        safe_name = recording_session.replace('/', '_')
        out_path = os.path.join(save_dir, f"timeseries_{safe_name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    except Exception:
        # Avoid raising plotting errors during evaluation; just skip plot
        try:
            plt.close('all')
        except Exception:
            pass
        return None
