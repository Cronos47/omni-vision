import time
import os

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from state import load_state
from utils import ema_smooth


STATE_PATH = "omnivision_state.json"

st.set_page_config(page_title="OmniVision", layout="wide")

# ---- STYLE ----
st.markdown(
    """
    <style>
    body { background-color: #0d0d0d; }
    .stApp { background-color: #0d0d0d; }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_slider(label, max_val, key):
    if max_val <= 0:
        return 0
    return st.slider(
        label,
        min_value=0,
        max_value=max_val,
        value=0,
        key=key,
    )


def safe_epoch_selector(epochs, label="Epoch", key="epoch_selector"):
    if not epochs:
        return None
    if len(epochs) == 1:
        return epochs[0]
    selected_epoch = st.slider(
        label,
        min_value=1,
        max_value=len(epochs),
        value=len(epochs),
        key=key,
    )
    return str(selected_epoch)


def get_state():
    if not os.path.exists(STATE_PATH):
        return None
    return load_state(STATE_PATH)


st.title("OmniVision")

state = get_state()

if state is None:
    st.warning("Waiting for state...")
    time.sleep(2)
    st.rerun()

# ---- LAYOUT ----
top_left, top_right = st.columns(2)
bottom_left, bottom_right = st.columns(2)

# =====================================================
# Q1 — PERFORMANCE RHYTHM
# =====================================================
with top_left:
    st.subheader("Performance Rhythm")

    metrics = state.get("history", {}).get("metrics", {})

    show_val = st.toggle("Show validation metrics", value=True)
    smoothing = st.slider("Smoothing", 0, 10, 0)

    fig = go.Figure()

    for metric_name, values in metrics.items():
        if not show_val and metric_name.startswith("val_"):
            continue

        y_vals = ema_smooth(values, smoothing)

        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(y_vals) + 1)),
                y=y_vals,
                mode="lines+markers",
                name=metric_name,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Q2 — LATENT GALAXY / DIFFUSION SUMMARY
# =====================================================
with top_right:
    arch_type = state.get("meta", {}).get("arch_type", "generic")

    if arch_type == "diffusion":
        st.subheader("Diffusion Summary")

        diffusion_epochs = sorted(state.get("diffusion", {}).keys(), key=lambda x: int(x)) if state.get("diffusion") else []
        diffusion_epoch_key = diffusion_epochs[-1] if diffusion_epochs else None

        if not diffusion_epoch_key:
            st.warning("No diffusion summary yet")
        else:
            diffusion_data = state.get("diffusion", {}).get(diffusion_epoch_key, {})
            energies = diffusion_data.get("energies", [])
            frames = diffusion_data.get("frames", [])

            if not energies:
                st.warning("No diffusion energy data available")
            else:
                start_energy = float(energies[0])
                end_energy = float(energies[-1])
                reduction = start_energy - end_energy
                reduction_pct = (reduction / (start_energy + 1e-8)) * 100.0

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Initial Noise", f"{start_energy:.4f}")
                with c2:
                    st.metric("Final Residual", f"{end_energy:.4f}", delta=f"{-reduction:.4f}")
                with c3:
                    st.metric("Denoise Gain", f"{reduction_pct:.1f}%")

                if frames:
                    final_frame = np.array(frames[-1])

                    fig = go.Figure(
                        data=go.Heatmap(
                            z=final_frame,
                            colorscale="Gray",
                            showscale=False,
                        )
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#1a1a1a",
                        plot_bgcolor="#1a1a1a",
                        height=420,
                        margin=dict(l=10, r=10, t=40, b=10),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

    else:
        st.subheader("Latent Galaxy")

        activations = state.get("activations", {})
        labels = state.get("labels", {})

        if not activations:
            st.warning("No activations yet")
        else:
            epochs = sorted(activations.keys(), key=lambda x: int(x))
            epoch_key = safe_epoch_selector(epochs, label="Epoch", key="latent_epoch_slider")

            epoch_acts = activations.get(epoch_key, {})

            if not epoch_acts:
                st.warning("No activations for this epoch")
            else:
                candidate_layer_name = None
                candidate_size = -1

                for lname, payload in epoch_acts.items():
                    vals = np.array(payload.get("values", []))
                    if payload.get("sampled", True) is False and vals.size > candidate_size:
                        candidate_layer_name = lname
                        candidate_size = vals.size

                if candidate_layer_name is None:
                    st.warning("No full activation layer available for Latent Galaxy")
                else:
                    data = epoch_acts[candidate_layer_name]["values"]
                    X = np.array(data)

                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                    elif X.ndim > 2:
                        X = X.reshape(X.shape[0], -1)

                    if X.shape[0] < 2:
                        st.warning("Need at least 2 samples for Latent Galaxy")
                    else:
                        if X.shape[1] < 3:
                            pad_width = 3 - X.shape[1]
                            X = np.pad(X, ((0, 0), (0, pad_width)), mode="constant")
                        elif X.shape[1] > 3:
                            if X.shape[0] < 200:
                                reducer = PCA(n_components=3)
                            else:
                                method = st.selectbox("Reducer", ["UMAP", "t-SNE"])
                                if method == "UMAP":
                                    reducer = umap.UMAP(n_components=3)
                                else:
                                    reducer = TSNE(n_components=3)
                            X = reducer.fit_transform(X)

                        y = labels.get(epoch_key, None)

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter3d(
                                x=X[:, 0],
                                y=X[:, 1],
                                z=X[:, 2],
                                mode="markers",
                                marker=dict(
                                    size=5,
                                    color=y if y is not None else list(range(X.shape[0])),
                                    colorscale="Viridis",
                                    opacity=0.8,
                                ),
                                text=[f"sample_{i}" for i in range(X.shape[0])],
                            )
                        )

                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#1a1a1a",
                            plot_bgcolor="#1a1a1a",
                            margin=dict(l=0, r=0, t=30, b=0),
                        )

                        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# Q3 — FEATURE / ATTENTION / SEQUENCE GRID
# =====================================================
with bottom_left:
    st.subheader("Feature / Attention / Sequence Grid")

    arch_type = state.get("meta", {}).get("arch_type", "generic")
    activations = state.get("activations", {})
    attention = state.get("attention", {})

    epochs = sorted(activations.keys(), key=lambda x: int(x)) if activations else []
    current_epoch_key = epochs[-1] if epochs else None

    # =========================
    # TRANSFORMER (unchanged)
    # =========================
    if arch_type == "transformer":
        st.info("Transformer view active (Sankey / Graph / Bar)")

        epoch_attention = attention.get(current_epoch_key, {})

        if not epoch_attention:
            st.warning("No attention payloads available")
        else:
            attention_layers = list(epoch_attention.keys())

            selected_layer = st.selectbox(
                "Attention Layer",
                options=attention_layers,
                index=0,
                key="attention_layer_select",
            )

            payload = epoch_attention[selected_layer]
            scores = np.array(payload.get("scores", []))

            if scores.ndim != 4:
                st.warning(f"Unexpected attention score shape: {scores.shape}")
            else:
                batch_size, num_heads, seq_len, _ = scores.shape

                sample_idx = safe_slider("Sample", batch_size - 1, "attention_sample_idx")

                head_idx = st.selectbox(
                    "Head",
                    options=list(range(num_heads)),
                    index=0,
                    key="attention_head_idx",
                )

                query_idx = st.selectbox(
                    "Query Token",
                    options=list(range(seq_len)),
                    index=0,
                    key="attention_query_idx",
                )

                attn = scores[sample_idx, head_idx]
                heatmap = attn

                fig = go.Figure(
                    data=go.Heatmap(
                        z=heatmap,
                        colorscale="Viridis",
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=500,
                )

                st.plotly_chart(fig, use_container_width=True)

    # =========================
    # CNN (unchanged)
    # =========================
    elif arch_type == "cnn":
        st.info("CNN feature map view active")

        epoch_acts = activations.get(current_epoch_key, {})

        conv_candidates = []
        for lname, payload in epoch_acts.items():
            vals = np.array(payload.get("values", []))
            if ("conv" in lname.lower()) and (payload.get("sampled", True) is False) and vals.ndim == 4:
                conv_candidates.append(lname)

        if not conv_candidates:
            st.warning("No full Conv activations available")
        else:
            selected_layer = st.selectbox(
                "Layer",
                options=conv_candidates,
                key="cnn_layer_select",
            )

            X = np.array(epoch_acts[selected_layer]["values"])

            sample_idx = safe_slider("Sample", X.shape[0] - 1, "cnn_sample")

            sample = X[sample_idx]
            _, _, C = sample.shape

            num_filters = min(C, 16)

            fig = make_subplots(rows=4, cols=4)

            for idx in range(num_filters):
                r = (idx // 4) + 1
                c = (idx % 4) + 1

                fig.add_trace(
                    go.Heatmap(
                        z=sample[:, :, idx],
                        colorscale="gray",
                        showscale=False,
                    ),
                    row=r,
                    col=c,
                )

            fig.update_layout(
                template="plotly_dark",
                height=650,
            )

            st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 🔥 SEQUENCE / HYBRID
    # =========================
    elif arch_type in ["rnn", "hybrid", "sequence_cnn"]:
        st.info("Sequence / Temporal view active")

        epoch_acts = activations.get(current_epoch_key, {})

        seq_candidates = []
        for lname, payload in epoch_acts.items():
            vals = np.array(payload.get("values", []))
            if (payload.get("sampled", True) is False) and vals.ndim == 3:
                seq_candidates.append(lname)

        if not seq_candidates:
            st.warning("No sequence activations found")
        else:
            selected_layer = st.selectbox(
                "Sequence Layer",
                options=seq_candidates,
                key="seq_layer_select",
            )

            X = np.array(epoch_acts[selected_layer]["values"])
            sample_idx = safe_slider("Sample", X.shape[0] - 1, "seq_sample")

            sample = X[sample_idx]  # [timesteps, hidden]

            view_mode = st.selectbox(
                "View",
                ["Heatmap", "Temporal Importance", "Trajectory", "Top Neurons"],
                key="seq_view_mode",
            )

            # =====================
            # Heatmap (existing)
            # =====================
            if view_mode == "Heatmap":
                if sample.shape[1] > 64:
                    sample = sample[:, :64]

                fig = go.Figure(
                    data=go.Heatmap(
                        z=sample,
                        colorscale="Viridis",
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=500,
                    xaxis_title="Hidden Units",
                    yaxis_title="Timesteps",
                )

                st.plotly_chart(fig, use_container_width=True)

            # =====================
            # 🔥 Temporal Importance
            # =====================
            elif view_mode == "Temporal Importance":
                importance = np.mean(np.abs(sample), axis=1)

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(importance))),
                        y=importance,
                        mode="lines+markers",
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=500,
                    xaxis_title="Timestep",
                    yaxis_title="Importance",
                )

                st.plotly_chart(fig, use_container_width=True)

                peak = int(np.argmax(importance))
                st.success(f"Most important timestep: {peak}")

            # =====================
            # 🔥 Trajectory (2D)
            # =====================
            elif view_mode == "Trajectory":
                from sklearn.decomposition import PCA

                if sample.shape[1] > 2:
                    reducer = PCA(n_components=2)
                    reduced = reducer.fit_transform(sample)
                else:
                    reduced = sample

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=reduced[:, 0],
                        y=reduced[:, 1],
                        mode="lines+markers+text",
                        text=[f"T{i}" for i in range(len(reduced))],
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=500,
                    xaxis_title="Component 1",
                    yaxis_title="Component 2",
                )

                st.plotly_chart(fig, use_container_width=True)

            # =====================
            # 🔥 Top Neurons
            # =====================
            else:
                neuron_scores = np.mean(np.abs(sample), axis=0)
                top_idx = np.argsort(neuron_scores)[-5:]

                fig = go.Figure()

                for i in top_idx:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(sample.shape[0])),
                            y=sample[:, i],
                            mode="lines",
                            name=f"N{i}",
                        )
                    )

                fig.update_layout(
                    template="plotly_dark",
                    height=500,
                    xaxis_title="Timestep",
                    yaxis_title="Activation",
                )

                st.plotly_chart(fig, use_container_width=True)

    elif arch_type == "diffusion":
        st.info("Diffusion Visualization")

        diffusion_epochs = sorted(state.get("diffusion", {}).keys(), key=lambda x: int(x)) if state.get("diffusion") else []
        diffusion_epoch_key = diffusion_epochs[-1] if diffusion_epochs else None

        if not diffusion_epoch_key:
            st.warning("No diffusion data available")
        else:
            diffusion_data = state.get("diffusion", {}).get(diffusion_epoch_key, {})

            if not diffusion_data:
                st.warning("No diffusion data available")
            else:
                frames = diffusion_data.get("frames", [])
                energies = diffusion_data.get("energies", [])
                embeddings = diffusion_data.get("embeddings", [])

                view_mode = st.selectbox(
                    "View",
                    ["Film Strip", "Noise Curve", "Latent Trajectory"],
                    key="diffusion_view_mode",
                )

                if view_mode == "Film Strip":
                    if not frames:
                        st.warning("No frames available")
                    else:
                        frame_arrays = [np.array(frame) for frame in frames]

                        base_frame = frame_arrays[0]

                        animated_fig = go.Figure(
                            data=[
                                go.Heatmap(
                                    z=base_frame,
                                    colorscale="Gray",
                                    showscale=False,
                                )
                            ],
                            frames=[
                                go.Frame(
                                    data=[
                                        go.Heatmap(
                                            z=frame_arrays[i],
                                            colorscale="Gray",
                                            showscale=False,
                                        )
                                    ],
                                    name=str(i),
                                )
                                for i in range(len(frame_arrays))
                            ],
                        )

                        animated_fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#1a1a1a",
                            plot_bgcolor="#1a1a1a",
                            height=520,
                            margin=dict(l=10, r=10, t=50, b=10),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            updatemenus=[
                                {
                                    "type": "buttons",
                                    "direction": "left",
                                    "x": 0.0,
                                    "y": 1.12,
                                    "showactive": True,
                                    "buttons": [
                                        {
                                            "label": "▶ Play",
                                            "method": "animate",
                                            "args": [
                                                None,
                                                {
                                                    "frame": {"duration": 350, "redraw": True},
                                                    "fromcurrent": True,
                                                    "transition": {"duration": 150},
                                                },
                                            ],
                                        },
                                        {
                                            "label": "⏸ Pause",
                                            "method": "animate",
                                            "args": [
                                                [None],
                                                {
                                                    "frame": {"duration": 0, "redraw": False},
                                                    "mode": "immediate",
                                                    "transition": {"duration": 0},
                                                },
                                            ],
                                        },
                                    ],
                                }
                            ],
                            sliders=[
                                {
                                    "active": 0,
                                    "y": -0.08,
                                    "x": 0.1,
                                    "len": 0.85,
                                    "currentvalue": {
                                        "prefix": "Timestep: ",
                                        "visible": True,
                                    },
                                    "steps": [
                                        {
                                            "label": str(i),
                                            "method": "animate",
                                            "args": [
                                                [str(i)],
                                                {
                                                    "mode": "immediate",
                                                    "frame": {"duration": 0, "redraw": True},
                                                    "transition": {"duration": 0},
                                                },
                                            ],
                                        }
                                        for i in range(len(frame_arrays))
                                    ],
                                }
                            ],
                        )

                        st.plotly_chart(
                            animated_fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )

                elif view_mode == "Noise Curve":
                    if not energies:
                        st.warning("No energy values available")
                    else:
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(len(energies))),
                                y=energies,
                                mode="lines+markers",
                                name="noise_energy",
                            )
                        )
                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#1a1a1a",
                            plot_bgcolor="#1a1a1a",
                            height=500,
                            xaxis_title="Timestep",
                            yaxis_title="Residual Noise Energy",
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )

                else:
                    if not embeddings:
                        st.warning("No latent embeddings available")
                    else:
                        from sklearn.decomposition import PCA

                        X = np.array(embeddings)

                        if X.ndim == 1:
                            X = X.reshape(-1, 1)

                        if X.shape[1] > 2:
                            X = PCA(n_components=2).fit_transform(X)
                        elif X.shape[1] == 1:
                            X = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=X[:, 0],
                                y=X[:, 1],
                                mode="lines+markers+text",
                                text=[f"t{i}" for i in range(len(X))],
                                textposition="top center",
                            )
                        )
                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#1a1a1a",
                            plot_bgcolor="#1a1a1a",
                            height=500,
                            xaxis_title="Component 1",
                            yaxis_title="Component 2",
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )

    # =========================
    # FALLBACK
    # =========================
    else:
        st.warning("Generic view — showing top neuron activations")

        epoch_acts = activations.get(current_epoch_key, {})
        if not epoch_acts:
            st.warning("No activations available")
        else:
            layer = list(epoch_acts.keys())[-1]
            vals = np.array(epoch_acts[layer]["values"])

            scores = np.mean(np.abs(vals), axis=0)
            top_idx = np.argsort(scores)[-20:]

            fig = go.Figure(
                data=go.Bar(
                    x=[f"N{i}" for i in top_idx],
                    y=scores[top_idx],
                )
            )

            fig.update_layout(template="plotly_dark")

            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Q4 — STRUCTURAL VITALITY
# =====================================================
# =====================================================
# Q4 — STRUCTURAL VITALITY / DIFFUSION VITALS
# =====================================================
with bottom_right:
    arch_type = state.get("meta", {}).get("arch_type", "generic")

    if arch_type == "diffusion":
        st.subheader("Diffusion Vitals")

        diffusion_epochs = sorted(state.get("diffusion", {}).keys(), key=lambda x: int(x)) if state.get("diffusion") else []
        diffusion_epoch_key = diffusion_epochs[-1] if diffusion_epochs else None

        if not diffusion_epoch_key:
            st.warning("No diffusion vitals available")
        else:
            diffusion_data = state.get("diffusion", {}).get(diffusion_epoch_key, {})
            energies = diffusion_data.get("energies", [])

            if not energies:
                st.warning("No energy values available")
            else:
                deltas = np.diff(energies) if len(energies) > 1 else np.array([0.0])
                avg_drop = float(-np.mean(deltas))
                smoothness = float(np.std(deltas)) if len(deltas) > 0 else 0.0
                converged = bool(energies[-1] < energies[0])

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Avg Denoise Step", f"{avg_drop:.4f}")
                with c2:
                    st.metric("Trajectory Smoothness", f"{smoothness:.4f}")
                with c3:
                    st.metric("Converging", "Yes" if converged else "No")

                bar_fig = go.Figure(
                    data=[
                        go.Bar(
                            x=[f"t{i}" for i in range(len(energies))],
                            y=energies,
                            text=[f"{v:.3f}" for v in energies],
                            textposition="outside",
                        )
                    ]
                )

                bar_fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#1a1a1a",
                    plot_bgcolor="#1a1a1a",
                    height=420,
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis_title="Diffusion Step",
                    yaxis_title="Residual Noise Energy",
                )

                st.plotly_chart(
                    bar_fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    else:
        st.subheader("Structural Vitality")

        gradients = state.get("gradients", {})

        if not gradients:
            st.warning("No gradients available")
        else:
            epochs = sorted(gradients.keys(), key=lambda x: int(x))
            epoch_key = epochs[-1]

            epoch_grads = gradients.get(epoch_key, {})

            if not epoch_grads:
                st.warning("No gradients for this epoch")
            else:
                layer_names = []
                grad_norms = []

                for layer_name, vars_dict in epoch_grads.items():
                    per_var_norms = []

                    for _, stats in vars_dict.items():
                        per_var_norms.append(float(stats.get("l2", 0.0)))

                    if len(per_var_norms) == 0:
                        normalized_norm = 0.0
                    else:
                        avg_norm = float(np.mean(per_var_norms))
                        normalized_norm = avg_norm / (len(per_var_norms) ** 0.5 + 1e-8)

                    layer_names.append(layer_name)
                    grad_norms.append(normalized_norm)

                colors = []
                for val in grad_norms:
                    if val > 0.1:
                        colors.append("red")
                    elif val < 1e-4:
                        colors.append("orange")
                    else:
                        colors.append("blue")

                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=layer_names,
                        y=grad_norms,
                        marker_color=colors,
                        text=[f"{v:.6f}" for v in grad_norms],
                        textposition="outside",
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#1a1a1a",
                    plot_bgcolor="#1a1a1a",
                    xaxis_title="Layer",
                    yaxis_title="Normalized Gradient Norm",
                    height=420,
                )

                st.plotly_chart(fig, use_container_width=True)

                exploding = [l for l, v in zip(layer_names, grad_norms) if v > 0.1]
                vanishing = [l for l, v in zip(layer_names, grad_norms) if v < 1e-4]

                if exploding:
                    st.error(f"Exploding gradients suspected in: {exploding}")
                elif vanishing:
                    st.warning(f"Vanishing gradients suspected in: {vanishing}")
                else:
                    st.success("Gradients look healthy")

# ---- AUTO REFRESH ----
time.sleep(2)
st.rerun()