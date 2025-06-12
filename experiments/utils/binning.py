import numpy as np

def get_bin_edges(confidence_scores, anomaly_scores, num_bins, bin_strategy):
    conf_bin_edges, anomaly_bin_edges = None, None
    final_conf_scores, final_anomaly_scores = None, None

    if bin_strategy == "quantile":
        # code is taken from https://github.com/mertyg/beyond-confidence-atypicality. The code available for use under the MIT license.
        quantiles_x = np.linspace(0.0, 1.0, num_bins + 1)
        quantiles_y = np.linspace(0.0, 1.0, num_bins + 1)
        confidence = np.max(confidence_scores, axis=1)
        anomaly_min = np.min(anomaly_scores, axis=1)

        anomaly_bin_edges = np.quantile(anomaly_min, q=quantiles_x)
        anomaly_bin_edges[0] = -np.inf
        anomaly_bin_edges[-1] = np.inf
        conf_bin_edges = np.quantile(confidence, q=quantiles_y)
        conf_bin_edges[0] = -np.inf
        conf_bin_edges[-1] = np.inf

        final_conf_scores = confidence
        final_anomaly_scores = anomaly_min

    elif bin_strategy == "uniform":
        confidence = np.max(confidence_scores, axis=1)
        conf_min, conf_max = confidence.min(), confidence.max()
        conf_bin_edges = np.linspace(conf_min, conf_max, num_bins + 1)

        anomaly = np.min(anomaly_scores, axis=1)
        anomaly_min, anomaly_max = anomaly.min(), anomaly.max()
        anomaly_bin_edges = np.linspace(anomaly_min, anomaly_max, num_bins + 1)

        final_conf_scores = confidence
        final_anomaly_scores = anomaly

    else:
        raise ValueError(f"Invalid value for bin_strategy: {bin_strategy}")
    
    return conf_bin_edges, anomaly_bin_edges, final_conf_scores, final_anomaly_scores