import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt
from scipy.stats import linregress

st.set_page_config(layout='wide')
st.title("Análise de Estabilograma da Plataforma de Força — CSV flexível")

# ====================== Funções auxiliares ======================


def infer_sep_and_read(file):
    """
    Tenta ler CSV inferindo separador. Tenta ',', ';' e '\\t'.
    Retorna DataFrame e o separador utilizado.
    """
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(file, sep=sep)
            if df.shape[1] >= 2:
                return df, sep
        except Exception:
            pass
    # Tenta inferência do pandas (engine python)
    try:
        df = pd.read_csv(file, sep=None, engine="python")
        return df, "infer"
    except Exception as e:
        raise RuntimeError(f"Falha ao ler CSV com separador variável: {e}")


def filtro_passabaixa(dados, fs, cutoff, ordem=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(ordem, normal_cutoff, btype='low')
    return filtfilt(b, a, dados)


def analise_difusao(cop_x, cop_y, fs, max_lag_seg=5):
    max_lag = int(max_lag_seg * fs)
    lags = np.arange(1, max_lag)
    msd_x = np.zeros_like(lags, dtype=float)
    msd_y = np.zeros_like(lags, dtype=float)
    for i, lag in enumerate(lags):
        dx = cop_x[lag:] - cop_x[:-lag]
        dy = cop_y[lag:] - cop_y[:-lag]
        msd_x[i] = np.mean(dx ** 2)
        msd_y[i] = np.mean(dy ** 2)
    msd_total = msd_x + msd_y
    return lags / fs, msd_total, msd_x, msd_y


def extrair_parametros_difusao(time_lags, msd_total):
    # janelas padrão (ajuste aqui se necessário)
    curto = time_lags <= 0.5
    longo = time_lags >= 1.5

    slope1, intercept1, *_ = linregress(time_lags[curto], msd_total[curto])
    slope2, intercept2, *_ = linregress(time_lags[longo], msd_total[longo])

    cp = (intercept2 - intercept1) / (slope1 - slope2)
    sway_cp = slope1 * cp + intercept1

    return {
        'open_loop_slope': slope1,
        'closed_loop_slope': slope2,
        'intercept1': intercept1,
        'intercept2': intercept2,
        'critical_point_time': cp,
        'sway_at_critical_point': sway_cp
    }


def plot_msd_total_com_regressao(time_lags, msd_total, parametros, ylim=(0, 70)):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time_lags, msd_total, label='MSD Total', color='black')

    t1 = time_lags[time_lags <= 0.5]
    t2 = time_lags[time_lags >= 1.5]

    ax.plot(t1, parametros['open_loop_slope'] * t1 + parametros['intercept1'],
            '--', color='green', label='Open-loop')
    ax.plot(t2, parametros['closed_loop_slope'] * t2 + parametros['intercept2'],
            '--', color='red', label='Closed-loop')

    ax.plot(parametros['critical_point_time'], parametros['sway_at_critical_point'],
            'ko', label='Critical Point')

    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("MSD (mm²)")
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig

# ====================== Upload e opções ======================


arquivo = st.file_uploader("Selecione o arquivo (.csv)", type=["csv"])

with st.expander("Opções de processamento", expanded=True):
    unidade_arquivo = st.selectbox(
        "Unidade das colunas COP no arquivo",
        options=["mm", "cm"],
        index=0,
        help="Se estiver em cm, será convertido para mm (×10)."
    )
    cutoff = st.number_input(
        "Frequência de corte do filtro passa-baixa (Hz)", 0.1, 50.0, value=5.0, step=0.1)
    t_ini = st.number_input(
        "Tempo inicial para análise (s)", 0.0, 1e6, value=30.0, step=0.5)
    t_fim = st.number_input("Tempo final para análise (s)",
                            0.0, 1e6, value=80.0, step=0.5)
    fs_override = st.number_input(
        "Taxa de amostragem (Hz) — deixe 0 para detectar por Tempo", 0.0, 5000.0, value=0.0, step=1.0)

st.markdown("---")

if arquivo is not None:
    # ---------- Leitura flexível ----------
    df, usado_sep = infer_sep_and_read(arquivo)
    st.caption(f"Separador detectado: **{usado_sep}**")
    st.dataframe(df.head(10), use_container_width=True)

    # ---------- Mapeamento de colunas ----------
    # Suporte a dois esquemas de nomes:
    # (A) 'Time (s)', 'COPx', 'COPy'
    # (B) 'Tempo', 'ML', 'AP'
    # Convenção adotada: COPx ≡ ML (médio-lateral), COPy ≡ AP (ântero-posterior)
    col_time = None
    if "Time (s)" in df.columns:
        col_time = "Time (s)"
    elif "Tempo" in df.columns:
        col_time = "Tempo"

    if "COPx" in df.columns and "COPy" in df.columns:
        col_x, col_y = "COPx", "COPy"
    elif "ML" in df.columns and "AP" in df.columns:
        col_x, col_y = "ML", "AP"
    else:
        st.error(
            "Não encontrei colunas de COP. Esperado: ('COPx','COPy') ou ('ML','AP').")
        st.stop()

    if col_time is None:
        st.error("Não encontrei a coluna de tempo. Esperado: 'Time (s)' ou 'Tempo'.")
        st.stop()

    # ---------- Normalização de tipos ----------
    # Força numérico e remove linhas inválidas
    for c in [col_time, col_x, col_y]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[col_time, col_x, col_y]).copy()

    # ---------- Unidades ----------
    # Converte para mm se arquivo estiver em cm
    if unidade_arquivo == "cm":
        df[col_x] = df[col_x] * 10.0
        df[col_y] = df[col_y] * 10.0
        st.info("Conversão aplicada: cm → mm (×10).")
    else:
        st.caption("Assumindo que as colunas já estão em mm.")

    # ---------- Definição de fs (Hz) ----------
    if fs_override > 0:
        fs = fs_override
    else:
        # Detecta pela coluna de tempo
        tempo = df[col_time].to_numpy()
        dts = np.diff(tempo)
        # usa mediana para robustez
        med_dt = np.median(dts[dts > 0]) if (dts > 0).any() else np.nan
        if np.isfinite(med_dt) and med_dt > 0:
            fs = 1.0 / med_dt
        else:
            st.error(
                "Não foi possível detectar a taxa de amostragem pelo tempo. Informe-a manualmente.")
            st.stop()

    st.success(f"Taxa de amostragem estimada: **{fs:.3f} Hz**" if fs_override ==
               0 else f"Taxa de amostragem definida: **{fs:.3f} Hz**")

    # ---------- Recorte por tempo ----------
    df_cut = df[(df[col_time] >= t_ini) & (df[col_time] <= t_fim)].copy()
    if df_cut.empty:
        st.error("Recorte de tempo resultou em DataFrame vazio. Ajuste t_ini/t_fim.")
        st.stop()

    # ---------- Detrend + Filtro ----------
    df_cut["COPx_detrended"] = detrend(df_cut[col_x].to_numpy())
    df_cut["COPy_detrended"] = detrend(df_cut[col_y].to_numpy())
    df_cut["COPx_filtrado"] = filtro_passabaixa(
        df_cut["COPx_detrended"], fs, cutoff)
    df_cut["COPy_filtrado"] = filtro_passabaixa(
        df_cut["COPy_detrended"], fs, cutoff)

    # ---------- Plots de séries ----------
    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(df_cut[col_time], df_cut["COPx_filtrado"],
                 label="Filtrado", linewidth=2, color='black')
        ax1.set_ylabel("COPx (mm)")
        ax1.set_xlabel("Tempo (s)")
        ax1.set_ylim(np.nanpercentile(df_cut["COPx_filtrado"], 1) - 10,
                     np.nanpercentile(df_cut["COPx_filtrado"], 99) + 10)
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(df_cut[col_time], df_cut["COPy_filtrado"],
                 label="Filtrado", linewidth=2, color='black')
        ax2.set_ylabel("COPy (mm)")
        ax2.set_xlabel("Tempo (s)")
        ax2.set_ylim(np.nanpercentile(df_cut["COPy_filtrado"], 1) - 10,
                     np.nanpercentile(df_cut["COPy_filtrado"], 99) + 10)
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    # ====================== Análise de Difusão ======================
    time_lags, msd_total, msd_x, msd_y = analise_difusao(
        df_cut['COPx_filtrado'].to_numpy(),
        df_cut['COPy_filtrado'].to_numpy(),
        fs=fs
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        parametros = extrair_parametros_difusao(time_lags, msd_total)
        st.pyplot(plot_msd_total_com_regressao(
            time_lags, msd_total, parametros))
        st.markdown(f"""
        - **Open-loop slope (α₁)**: {parametros['open_loop_slope']:.4f} mm²/s  
        - **Closed-loop slope (α₂)**: {parametros['closed_loop_slope']:.4f} mm²/s  
        - **Critical Point**: {parametros['critical_point_time']:.4f} s  
        - **Open-loop sway**: {parametros['sway_at_critical_point']:.4f} mm²
        """)

    with c2:
        parametros_x = extrair_parametros_difusao(time_lags, msd_x)
        st.pyplot(plot_msd_total_com_regressao(
            time_lags, msd_x, parametros_x, ylim=None))
        st.markdown(f"""
        - **Open-loop slope X (α₁)**: {parametros_x['open_loop_slope']:.4f} mm²/s  
        - **Closed-loop slope X (α₂)**: {parametros_x['closed_loop_slope']:.4f} mm²/s  
        - **Critical Point X**: {parametros_x['critical_point_time']:.4f} s  
        - **Open-loop sway X**: {parametros_x['sway_at_critical_point']:.4f} mm²
        """)

    with c3:
        parametros_y = extrair_parametros_difusao(time_lags, msd_y)
        st.pyplot(plot_msd_total_com_regressao(
            time_lags, msd_y, parametros_y, ylim=None))
        st.markdown(f"""
        - **Open-loop slope Y (α₁)**: {parametros_y['open_loop_slope']:.4f} mm²/s  
        - **Closed-loop slope Y (α₂)**: {parametros_y['closed_loop_slope']:.4f} mm²/s  
        - **Critical Point Y**: {parametros_y['critical_point_time']:.4f} s  
        - **Open-loop sway Y**: {parametros_y['sway_at_critical_point']:.4f} mm²
        """)

    # ====================== Exportações ======================
    st.subheader("Exportar dados da análise de difusão")
    df_diffusao = pd.DataFrame({
        "Lag (s)": time_lags,
        "MSD Total (mm²)": msd_total,
        "MSD COPx (mm²)": msd_x,
        "MSD COPy (mm²)": msd_y
    })
    st.download_button(
        label="📥 Baixar CSV da Análise de Difusão",
        data=df_diffusao.to_csv(index=False).encode('utf-8'),
        file_name='analise_difusao.csv',
        mime='text/csv'
    )

    st.subheader("Exportar parâmetros da análise de difusão")
    df_parametros = pd.DataFrame({
        "Parâmetro": [
            "Open-loop slope global (α₁)",
            "Closed-loop slope global (α₂)",
            "Critical Point global (s)",
            "Sway no Critical Point global (mm²)",
            "Open-loop slope X (α₁)",
            "Closed-loop slope X (α₂)",
            "Critical Point X (s)",
            "Sway no Critical Point X (mm²)",
            "Open-loop slope Y (α₁)",
            "Closed-loop slope Y (α₂)",
            "Critical Point Y (s)",
            "Sway no Critical Point Y (mm²)"
        ],
        "Valor": [
            parametros['open_loop_slope'],
            parametros['closed_loop_slope'],
            parametros['critical_point_time'],
            parametros['sway_at_critical_point'],
            parametros_x['open_loop_slope'],
            parametros_x['closed_loop_slope'],
            parametros_x['critical_point_time'],
            parametros_x['sway_at_critical_point'],
            parametros_y['open_loop_slope'],
            parametros_y['closed_loop_slope'],
            parametros_y['critical_point_time'],
            parametros_y['sway_at_critical_point']
        ]
    })
    st.download_button(
        label="📥 Baixar CSV dos Parâmetros Extraídos",
        data=df_parametros.to_csv(index=False).encode('utf-8'),
        file_name='parametros_difusao.csv',
        mime='text/csv'
    )
