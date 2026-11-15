import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import (
    binom, poisson, nbinom, norm, expon, gamma as gamma_dist,
    beta as beta_dist, cauchy, uniform, geom, chi2,
    t as t_student, f as f_fisher
)
from scipy.special import gamma as gamma_func, beta as beta_func

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="BayesLab - Estatística Bayesiana e Clássica",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Suíte de Estatística Bayesiana & Distribuições")

# ==============================================================================
# FUNÇÃO PADRÃO DE RELATÓRIO E GRÁFICO PARA MODELOS CONJUGADOS CONTÍNUOS
# ==============================================================================
def exibir_relatorio_conjugado(dist_priori, dist_post, nome_priori, nome_post,
                               x_vals, x1_val, x2_val, casas_dec, titulo_graf, xlabel_graf,
                               simbolo_param="θ", info_extra=""):
    """Exibe parâmetros, tabela comparativa Priori vs Posteriori e gráfico com intervalos."""
    a, b = sorted((x1_val, x2_val))
    fmt = f"{{:.{casas_dec}f}}"

    # Probabilidades na Priori
    p_pri_menor_x1 = dist_priori.cdf(x1_val)
    p_pri_maior_x1 = dist_priori.sf(x1_val)
    p_pri_menor_x2 = dist_priori.cdf(x2_val)
    p_pri_maior_x2 = dist_priori.sf(x2_val)
    p_pri_interv   = dist_priori.cdf(b) - dist_priori.cdf(a)

    # Probabilidades na Posteriori
    p_pos_menor_x1 = dist_post.cdf(x1_val)
    p_pos_maior_x1 = dist_post.sf(x1_val)
    p_pos_menor_x2 = dist_post.cdf(x2_val)
    p_pos_maior_x2 = dist_post.sf(x2_val)
    p_pos_interv   = dist_post.cdf(b) - dist_post.cdf(a)

    # Resumo em destaque
    c1, c2 = st.columns(2)
    c1.info(f"**Priori (Antes):** `{nome_priori}`")
    c2.error(f"**Posteriori (Depois):** `{nome_post}`")
    if info_extra:
        st.caption(f"ℹ️ {info_extra}")

    # Tabela comparativa
    dados_tabela = [
        [f"P({simbolo_param} < {x1_val:g})", fmt.format(p_pri_menor_x1), fmt.format(p_pos_menor_x1)],
        [f"P({simbolo_param} > {x1_val:g})", fmt.format(p_pri_maior_x1), fmt.format(p_pos_maior_x1)],
        [f"P({simbolo_param} < {x2_val:g})", fmt.format(p_pri_menor_x2), fmt.format(p_pos_menor_x2)],
        [f"P({simbolo_param} > {x2_val:g})", fmt.format(p_pri_maior_x2), fmt.format(p_pos_maior_x2)],
        [f"P({a:g} < {simbolo_param} < {b:g})", fmt.format(p_pri_interv), fmt.format(p_pos_interv)],
        [f"Média E[{simbolo_param}]", fmt.format(dist_priori.mean()), fmt.format(dist_post.mean())],
        [f"Variância Var[{simbolo_param}]", fmt.format(dist_priori.var()), fmt.format(dist_post.var())],
        [f"Desvio Padrão DP[{simbolo_param}]", fmt.format(dist_priori.std()), fmt.format(dist_post.std())]
    ]
    df_res = pd.DataFrame(
        dados_tabela,
        columns=["Medida / Probabilidade", "Priori (Antes)", "Posteriori (Depois)"]
    )

    col_tab, col_graf = st.columns([1, 1.4])
    with col_tab:
        st.subheader("📋 Tabela Comparativa")
        st.table(df_res.set_index("Medida / Probabilidade"))

    with col_graf:
        st.subheader("📈 Gráfico das Distribuições")
        priori_pdf = dist_priori.pdf(x_vals)
        post_pdf = dist_post.pdf(x_vals)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(x_vals, priori_pdf, label=f'Priori: {nome_priori}', color='#3498db', linestyle='--', linewidth=1.8)
        ax.plot(x_vals, post_pdf, label=f'Posteriori: {nome_post}', color='#e74c3c', linewidth=2.2)

        mascara = (x_vals >= a) & (x_vals <= b)

        # Intervalo da Priori exatamente em cima do da Posteriori
        ax.fill_between(
            x_vals, priori_pdf, where=mascara, color='#3498db', alpha=0.18,
            label=f'Priori P({a:g} < {simbolo_param} < {b:g}) = {fmt.format(p_pri_interv)}'
        )
        ax.fill_between(
            x_vals, post_pdf, where=mascara, color='#e74c3c', alpha=0.25,
            label=f'Posteriori P({a:g} < {simbolo_param} < {b:g}) = {fmt.format(p_pos_interv)}'
        )

        ax.axvline(x=x1_val, linestyle=":", color='darkred', alpha=0.8, label=f'x1 = {x1_val:g}')
        ax.axvline(x=x2_val, linestyle=":", color='darkgreen', alpha=0.8, label=f'x2 = {x2_val:g}')

        ax.set_title(titulo_graf, fontsize=13)
        ax.set_xlabel(xlabel_graf, fontsize=11)
        ax.set_ylabel('Densidade', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ==============================================================================
# ABAS PRINCIPAIS DO APLICATIVO
# ==============================================================================
abas_principais = st.tabs([
    "1. Teorema de Bayes",
    "2. Beta-Binomial",
    "3. Beta-Geométrica",
    "4. Beta-Bin. Negativa",
    "5. Gama-Poisson",
    "6. Gama-Exponencial",
    "7. Explorador de Distribuições"
])

# ==============================================================================
# ABA 1: TEOREMA DE BAYES DISCRETO
# ==============================================================================
with abas_principais[0]:
    st.subheader("Teorema de Bayes - Atualização de Probabilidades (Discreto)")

    c1, c2, c3, c4, c5 = st.columns([1, 1.5, 1.5, 1.2, 1])
    num_estados = c1.number_input("Nº Opções:", min_value=2, max_value=10, value=2, step=1)
    nome_evento = c2.text_input("Evento (Evidência - B):", value="Recessão")
    termo_a = c3.text_input("Termo p/ A (ex: Premissa):", value="Premissa")
    simbolo_a = c4.text_input("Símbolo p/ A (ex: A, H):", value="A")
    casas_disc = c5.number_input("Casas decimais:", min_value=2, max_value=10, value=4, step=1)

    ev = nome_evento.strip() if nome_evento.strip() else "B"
    t_a = termo_a.strip() if termo_a.strip() else "Opção"
    s_a = simbolo_a.strip() if simbolo_a.strip() else "A"
    fmt_disc = f"{{:.{casas_disc}f}}"
    fmt_pct = f"{{:.{max(2, casas_disc - 2)}f}}%"

    st.markdown("##### Defina as Probabilidades a Priori e as Verossimilhanças:")
    estados_dict = {}

    for i in range(num_estados):
        col_n, col_p, col_v = st.columns([1.5, 1.5, 1.5])
        padrao_p = 0.80 if i == 0 else (0.20 if i == 1 else 0.0)
        padrao_c = 0.05 if i == 0 else (0.50 if i == 1 else 0.0)

        nome_input = col_n.text_input(
            f"{t_a} #{i+1}:", value="", placeholder=f"{s_a}{i+1}", key=f"disc_nome_{i}"
        )
        nome_val = nome_input.strip() if nome_input.strip() else f"{s_a}{i+1}"
        if nome_val in estados_dict:
            nome_val = f"{nome_val} ({i+1})"

        priori_val = col_p.number_input(
            f"P({nome_val}) [Priori]:", min_value=0.0, max_value=1.0, value=padrao_p,
            step=0.01, format="%.4f", key=f"disc_p_{i}"
        )
        veros_val = col_v.number_input(
            f"P({ev} | {nome_val}) [Verossim.]:", min_value=0.0, max_value=1.0, value=padrao_c,
            step=0.01, format="%.4f", key=f"disc_v_{i}"
        )
        estados_dict[nome_val] = [priori_val, veros_val]

    nomes_lista = list(estados_dict.keys())
    soma_priors = sum(estados_dict[k][0] for k in nomes_lista)
    p_ev = sum(estados_dict[k][0] * estados_dict[k][1] for k in nomes_lista)

    st.divider()
    col_sel, col_res1, col_res2 = st.columns([1.2, 1.4, 1.4])
    alvo = col_sel.selectbox("🎯 Destacar Posteriori de:", options=nomes_lista)

    if not np.isclose(soma_priors, 1.0, atol=0.01):
        st.warning(f"⚠️ Atenção: A soma das probabilidades a priori é **{fmt_disc.format(soma_priors)}** (deveria ser 1.0).")

    if p_ev == 0:
        st.error(f"❌ Erro: A probabilidade total do evento P({ev}) é 0.")
    else:
        priori_alvo, veros_alvo = estados_dict[alvo]
        posteriori_alvo = (priori_alvo * veros_alvo) / p_ev

        col_res1.metric(
            label=f"Probabilidade Total P({ev})",
            value=f"{fmt_disc.format(p_ev)} ({fmt_pct.format(p_ev*100)})"
        )
        col_res2.metric(
            label=f"Posteriori em Destaque P({alvo} | {ev})",
            value=f"{fmt_disc.format(posteriori_alvo)} ({fmt_pct.format(posteriori_alvo*100)})"
        )

        sub_disc = st.tabs(["📋 Tabela de Bayes", "📊 Gráfico de Barras", "🌳 Diagrama de Árvore"])

        tabela_dados = []
        priors_lista = []
        posteriors_lista = []

        for nome in nomes_lista:
            pri, ver = estados_dict[nome]
            conj = pri * ver
            post = conj / p_ev
            priors_lista.append(pri)
            posteriors_lista.append(post)
            tabela_dados.append([
                nome,
                fmt_disc.format(pri),
                fmt_disc.format(ver),
                fmt_disc.format(conj),
                f"{fmt_disc.format(post)} ({fmt_pct.format(post*100)})"
            ])
        tabela_dados.append(["TOTAL", fmt_disc.format(soma_priors), "-", fmt_disc.format(p_ev), "1.0000 (100%)"])

        with sub_disc[0]:
            df_disc = pd.DataFrame(
                tabela_dados,
                columns=[t_a, f"P({t_a}) [Priori]", f"P({ev} | {t_a}) [Verossim.]", "Conjunta", f"P({t_a} | {ev}) [Posteriori]"]
            )
            st.table(df_disc.set_index(t_a))

        with sub_disc[1]:
            fig_bar, ax_bar = plt.subplots(figsize=(8, 4.2))
            x_pos = np.arange(len(nomes_lista))
            width = 0.35
            ax_bar.bar(x_pos - width/2, priors_lista, width, label='A Priori', color='#95a5a6', edgecolor='black')
            ax_bar.bar(x_pos + width/2, posteriors_lista, width, label=f'A Posteriori (Dado {ev})', color='#2980b9', edgecolor='black')
            ax_bar.set_ylabel('Probabilidade')
            ax_bar.set_title(f'Atualização Bayesiana após observar: "{ev}"')
            ax_bar.set_xticks(x_pos)
            ax_bar.set_xticklabels(nomes_lista)
            ax_bar.set_ylim(0, max(max(priors_lista), max(posteriors_lista)) * 1.25)
            ax_bar.legend()
            ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
            for i, val in enumerate(priors_lista):
                ax_bar.text(i - width/2, val + 0.01, f"{val:.3f}", ha='center', va='bottom', fontsize=9)
            for i, val in enumerate(posteriors_lista):
                ax_bar.text(i + width/2, val + 0.01, f"{val:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2980b9')
            fig_bar.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        with sub_disc[2]:
            fig_tree, ax_tree = plt.subplots(figsize=(11, max(4.5, num_estados * 1.6)))
            ax_tree.axis('off')
            ax_tree.text(0, 0.5, 'Início', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray'))

            y_step = 1.0 / (num_estados + 1)
            for i, nome in enumerate(nomes_lista):
                y_a = 1.0 - (i + 1) * y_step
                pri, ver = estados_dict[nome]
                prob_nao_ev = 1.0 - ver

                ax_tree.plot([0.1, 0.9], [0.5, y_a], 'k-', lw=1.5)
                ax_tree.text(0.5, (0.5 + y_a)/2 + 0.02, f"P({nome})={pri:.4g}", ha='center', va='bottom', fontsize=9.5, color='darkblue')
                ax_tree.text(1.0, y_a, nome, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', ec='blue'))

                y_up = y_a + y_step / 3.5
                y_down = y_a - y_step / 3.5

                ax_tree.plot([1.1, 2.3], [y_a, y_up], 'k-', lw=1.5)
                ax_tree.text(1.7, (y_a + y_up)/2 + 0.02, f"P({ev}|{nome})={ver:.4g}", ha='center', va='bottom', fontsize=9, color='green')
                ax_tree.text(2.4, y_up, ev, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.4', fc='lightgreen', ec='green'))
                ax_tree.text(2.65, y_up, f"∩ = {fmt_disc.format(pri * ver)}", ha='left', va='center', fontsize=9.5, fontweight='bold')

                ax_tree.plot([1.1, 2.3], [y_a, y_down], 'k--', lw=1.5, alpha=0.6)
                ax_tree.text(1.7, (y_a + y_down)/2 - 0.02, f"P(Não {ev}|{nome})={prob_nao_ev:.4g}", ha='center', va='top', fontsize=9, color='red')
                ax_tree.text(2.4, y_down, f"Não {ev}", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.4', fc='#ffcccc', ec='red'))
                ax_tree.text(2.65, y_down, f"∩ = {fmt_disc.format(pri * prob_nao_ev)}", ha='left', va='center', fontsize=9.5, alpha=0.6)

            ax_tree.set_xlim(-0.2, 3.5)
            ax_tree.set_ylim(0, 1)
            ax_tree.set_title(f'Diagrama de Árvore: Probabilidade Total P({ev})', fontsize=13, pad=15)
            fig_tree.tight_layout()
            st.pyplot(fig_tree)
            plt.close(fig_tree)


# ==============================================================================
# ABA 2: MODELO BETA-BINOMIAL
# ==============================================================================
with abas_principais[1]:
    st.subheader("Modelo Beta-Binomial (Proporções em 'n' ensaios fixos)")
    c1, c2, c3, c4 = st.columns(4)
    bb_a = c1.number_input("Priori 'a':", min_value=0.0001, value=4.0, step=0.5, key="bb_a")
    bb_b = c2.number_input("Priori 'b':", min_value=0.0001, value=2.0, step=0.5, key="bb_b")
    bb_n = c3.number_input("Ensaios (n):", min_value=1, value=4, step=1, key="bb_n")
    bb_k = c4.number_input("Sucessos (k):", min_value=0, value=3, step=1, key="bb_k")

    c5, c6, c7 = st.columns(3)
    bb_x1 = c5.number_input("Limite x1:", min_value=0.0, max_value=1.0, value=0.50, step=0.05, format="%.4f", key="bb_x1")
    bb_x2 = c6.number_input("Limite x2:", min_value=0.0, max_value=1.0, value=0.80, step=0.05, format="%.4f", key="bb_x2")
    bb_casas = c7.number_input("Casas decimais:", min_value=1, max_value=10, value=5, step=1, key="bb_casas")

    if bb_k > bb_n:
        st.error("❌ Erro: O número de sucessos (k) não pode ser maior que o de ensaios (n).")
    else:
        a_post = bb_a + bb_k
        b_post = bb_b + bb_n - bb_k
        d_pri = beta_dist(bb_a, bb_b)
        d_pos = beta_dist(a_post, b_post)
        theta = np.linspace(0, 1, 500)
        int_norm = beta_func(a_post, b_post) / beta_func(bb_a, bb_b)
        info = f"Sucessos: {bb_k} | Fracassos: {bb_n - bb_k} | Integral do Núcleo: {int_norm:.6e}"

        exibir_relatorio_conjugado(
            d_pri, d_pos, f"Beta({bb_a:g}; {bb_b:g})", f"Beta({a_post:g}; {b_post:g})",
            theta, bb_x1, bb_x2, bb_casas,
            "Atualização Bayesiana (Modelo Beta-Binomial)", r"$\theta$ (Probabilidade de Sucesso)", "θ", info
        )


# ==============================================================================
# ABA 3: MODELO BETA-GEOMÉTRICA
# ==============================================================================
with abas_principais[2]:
    st.subheader("Modelo Beta-Geométrica (Ensaios até o 1º sucesso por sequência)")
    c1, c2, c3, c4 = st.columns(4)
    bg_a = c1.number_input("Priori 'a':", min_value=0.0001, value=6.0, step=0.5, key="bg_a")
    bg_b = c2.number_input("Priori 'b':", min_value=0.0001, value=9.0, step=0.5, key="bg_b")
    bg_n = c3.number_input("Total de Ensaios (n):", min_value=1, value=28, step=1, key="bg_n")
    bg_m = c4.number_input("Sequências (sucessos):", min_value=1, value=12, step=1, key="bg_m")

    c5, c6, c7 = st.columns(3)
    bg_x1 = c5.number_input("Limite x1:", min_value=0.0, max_value=1.0, value=0.25, step=0.05, format="%.4f", key="bg_x1")
    bg_x2 = c6.number_input("Limite x2:", min_value=0.0, max_value=1.0, value=0.62, step=0.05, format="%.4f", key="bg_x2")
    bg_casas = c7.number_input("Casas decimais:", min_value=1, max_value=10, value=5, step=1, key="bg_casas")

    if bg_m > bg_n:
        st.error("❌ Erro: O número de sequências não pode ser maior que o total de ensaios (n).")
    else:
        fracassos = bg_n - bg_m
        a_post = bg_a + bg_m
        b_post = bg_b + fracassos
        d_pri = beta_dist(bg_a, bg_b)
        d_pos = beta_dist(a_post, b_post)
        theta = np.linspace(0, 1, 500)
        info = f"Sucessos (sequências): {bg_m} | Fracassos (n - m): {fracassos}"

        exibir_relatorio_conjugado(
            d_pri, d_pos, f"Beta({bg_a:g}; {bg_b:g})", f"Beta({a_post:g}; {b_post:g})",
            theta, bg_x1, bg_x2, bg_casas,
            "Atualização Bayesiana (Modelo Beta-Geométrica)", r"$\theta$ (Probabilidade de Sucesso)", "θ", info
        )


# ==============================================================================
# ABA 4: MODELO BETA-BINOMIAL NEGATIVA
# ==============================================================================
with abas_principais[3]:
    st.subheader("Modelo Beta-Binomial Negativa (Ensaios até atingir 'r' sucessos)")
    c1, c2, c3, c4 = st.columns(4)
    bbn_a = c1.number_input("Priori 'a':", min_value=0.0001, value=12.0, step=0.5, key="bbn_a")
    bbn_b = c2.number_input("Priori 'b':", min_value=0.0001, value=7.0, step=0.5, key="bbn_b")
    bbn_n = c3.number_input("Total de Ensaios (n):", min_value=1, value=24, step=1, key="bbn_n")
    bbn_r = c4.number_input("Sucessos fixos (r):", min_value=1, value=9, step=1, key="bbn_r")

    c5, c6, c7 = st.columns(3)
    bbn_x1 = c5.number_input("Limite x1:", min_value=0.0, max_value=1.0, value=0.30, step=0.05, format="%.4f", key="bbn_x1")
    bbn_x2 = c6.number_input("Limite x2:", min_value=0.0, max_value=1.0, value=0.75, step=0.05, format="%.4f", key="bbn_x2")
    bbn_casas = c7.number_input("Casas decimais:", min_value=1, max_value=10, value=5, step=1, key="bbn_casas")

    if bbn_r > bbn_n:
        st.error("❌ Erro: O número de sucessos fixados (r) não pode ser maior que o total de ensaios (n).")
    else:
        fracassos = bbn_n - bbn_r
        a_post = bbn_a + bbn_r
        b_post = bbn_b + fracassos
        d_pri = beta_dist(bbn_a, bbn_b)
        d_pos = beta_dist(a_post, b_post)
        theta = np.linspace(0, 1, 500)
        info = f"Sucessos fixos (r): {bbn_r} | Fracassos (n - r): {fracassos}"

        exibir_relatorio_conjugado(
            d_pri, d_pos, f"Beta({bbn_a:g}; {bbn_b:g})", f"Beta({a_post:g}; {b_post:g})",
            theta, bbn_x1, bbn_x2, bbn_casas,
            "Atualização Bayesiana (Modelo Beta-Binomial Negativa)", r"$\theta$ (Probabilidade de Sucesso)", "θ", info
        )


# ==============================================================================
# ABA 5: MODELO GAMA-POISSON
# ==============================================================================
with abas_principais[4]:
    st.subheader("Modelo Gama-Poisson (Taxas de Contagem)")
    c0, c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1, 1])
    gp_param = c0.selectbox("Parametrização:", ["Escala (β)", "Taxa (β)"], key="gp_param")
    gp_alpha = c1.number_input("Priori alfa (α):", min_value=0.0001, value=3.0, step=0.5, format="%.4f", key="gp_a")
    gp_beta = c2.number_input("Priori beta (β):", min_value=0.0001, value=1.0, step=0.1, format="%.4f", key="gp_b")
    gp_n = c3.number_input("Períodos (n):", min_value=1, value=4, step=1, key="gp_n")
    gp_soma = c4.number_input("Soma (Σx):", min_value=0, value=10, step=1, key="gp_soma")

    c5, c6, c7 = st.columns(3)
    gp_x1 = c5.number_input("Limite x1:", value=1.50, step=0.1, format="%.4f", key="gp_x1")
    gp_x2 = c6.number_input("Limite x2:", value=3.70, step=0.1, format="%.4f", key="gp_x2")
    gp_casas = c7.number_input("Casas decimais:", min_value=1, max_value=10, value=5, step=1, key="gp_casas")

    a_post = gp_alpha + gp_soma
    if gp_param == "Escala (β)":
        b_post = 1.0 / ((1.0 / gp_beta) + gp_n)
        scale_pri, scale_pos = gp_beta, b_post
        rotulo_tipo = "escala"
    else:
        b_post = gp_beta + gp_n
        scale_pri, scale_pos = 1.0 / gp_beta, 1.0 / b_post
        rotulo_tipo = "taxa"

    d_pri = gamma_dist(a=gp_alpha, scale=scale_pri)
    d_pos = gamma_dist(a=a_post, scale=scale_pos)

    int_nucleo = (gamma_func(a_post) * (scale_pos ** a_post)) / (gamma_func(gp_alpha) * (scale_pri ** gp_alpha))
    info = f"Parametrização por {rotulo_tipo} | Integral de Normalização (núcleo): {int_nucleo:.6f}"
    if gp_n == 1:
        int_completa = int_nucleo / gamma_func(gp_soma + 1)
        info += f" | Com 1/x!: {int_completa:.6f}"

    limite_sup = max(d_pri.ppf(0.995), d_pos.ppf(0.995), gp_x1 * 1.2, gp_x2 * 1.2, 0.1)
    lambda_vals = np.linspace(0, limite_sup, 500)

    exibir_relatorio_conjugado(
        d_pri, d_pos, f"Gama({gp_alpha:g}; {gp_beta:g})", f"Gama({a_post:g}; {b_post:.6f})",
        lambda_vals, gp_x1, gp_x2, gp_casas,
        "Atualização Bayesiana (Modelo Gama-Poisson)", r"$\lambda$ (Taxa de Ocorrência)", "λ", info
    )


# ==============================================================================
# ABA 6: MODELO GAMA-EXPONENCIAL
# ==============================================================================
with abas_principais[5]:
    st.subheader("Modelo Gama-Exponencial (Tempos até Evento)")
    c0, c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1, 1])
    ge_param = c0.selectbox("Parametrização:", ["Escala (β)", "Taxa (β)"], key="ge_param")
    ge_alpha = c1.number_input("Priori alfa (α):", min_value=0.0001, value=3.0, step=0.5, format="%.4f", key="ge_a")
    ge_beta = c2.number_input("Priori beta (β):", min_value=0.0001, value=1.0, step=0.1, format="%.4f", key="ge_b")
    ge_n = c3.number_input("Observações (n):", min_value=1, value=5, step=1, key="ge_n")
    ge_soma = c4.number_input("Soma dos tempos (Σx):", min_value=0.0001, value=5.0, step=0.5, format="%.4f", key="ge_soma")

    c5, c6, c7 = st.columns(3)
    ge_x1 = c5.number_input("Limite x1:", value=0.70, step=0.1, format="%.4f", key="ge_x1")
    ge_x2 = c6.number_input("Limite x2:", value=1.60, step=0.1, format="%.4f", key="ge_x2")
    ge_casas = c7.number_input("Casas decimais:", min_value=1, max_value=10, value=5, step=1, key="ge_casas")

    a_post = ge_alpha + ge_n
    if ge_param == "Escala (β)":
        b_post = 1.0 / ((1.0 / ge_beta) + ge_soma)
        scale_pri, scale_pos = ge_beta, b_post
        rotulo_tipo = "escala"
    else:
        b_post = ge_beta + ge_soma
        scale_pri, scale_pos = 1.0 / ge_beta, 1.0 / b_post
        rotulo_tipo = "taxa"

    d_pri = gamma_dist(a=ge_alpha, scale=scale_pri)
    d_pos = gamma_dist(a=a_post, scale=scale_pos)

    int_norm = (gamma_func(a_post) * (scale_pos ** a_post)) / (gamma_func(ge_alpha) * (scale_pri ** ge_alpha))
    info = f"Parametrização por {rotulo_tipo} | Integral de Normalização: {int_norm:.6f}"

    limite_sup = max(d_pri.ppf(0.995), d_pos.ppf(0.995), ge_x1 * 1.2, ge_x2 * 1.2, 0.1)
    lambda_vals = np.linspace(0, limite_sup, 500)

    exibir_relatorio_conjugado(
        d_pri, d_pos, f"Gama({ge_alpha:g}; {ge_beta:g})", f"Gama({a_post:g}; {b_post:.6f})",
        lambda_vals, ge_x1, ge_x2, ge_casas,
        "Atualização Bayesiana (Modelo Gama-Exponencial)", r"$\lambda$ (Taxa de Falha / Tempo)", "λ", info
    )


# ==============================================================================
# ABA 7: EXPLORADOR DE DISTRIBUIÇÕES (ESTATÍSTICA CLÁSSICA)
# ==============================================================================
with abas_principais[6]:
    st.subheader("Explorador de Distribuições de Probabilidade")

    col_dist, col_p1, col_p2 = st.columns([1.4, 1, 1])
    dist_escolhida = col_dist.selectbox(
        "Selecione a Distribuição:",
        [
            "Binomial", "Poisson", "Binomial Negativa", "Geométrica",
            "Normal", "Uniforme Contínua", "Exponencial", "Gama",
            "Beta", "Qui-quadrado", "t-Student", "F", "Cauchy"
        ]
    )

    is_discrete = False
    suporte_finito = False

    if dist_escolhida == "Binomial":
        n_val = col_p1.number_input("n (Ensaios):", min_value=1, value=10, step=1)
        p_val = col_p2.number_input("p (Sucesso):", min_value=0.0, max_value=1.0, value=0.30, step=0.05, format="%.4f")
        dist = binom(n=n_val, p=p_val)
        nome_dist = f"Binomial(n={n_val}, p={p_val:g})"
        is_discrete, suporte_finito = True, True
        suporte_min, suporte_max = 0, n_val

    elif dist_escolhida == "Poisson":
        lam_val = col_p1.number_input("λ (Taxa):", min_value=0.0001, value=4.0, step=0.5, format="%.4f")
        xmax_val = col_p2.number_input("x_max (Gráfico/Tabela):", min_value=1, value=20, step=1)
        dist = poisson(mu=lam_val)
        nome_dist = f"Poisson(λ={lam_val:g})"
        is_discrete = True
        suporte_min, suporte_max = 0, xmax_val

    elif dist_escolhida == "Binomial Negativa":
        r_val = col_p1.number_input("r (Sucessos):", min_value=1, value=5, step=1)
        p_val = col_p2.number_input("p (Sucesso):", min_value=0.0001, max_value=1.0, value=0.30, step=0.05, format="%.4f")
        xmax_val = st.number_input("x_max (Gráfico/Tabela):", min_value=1, value=25, step=1)
        dist = nbinom(n=r_val, p=p_val)
        nome_dist = f"Binomial Negativa(r={r_val}, p={p_val:g})"
        is_discrete = True
        suporte_min, suporte_max = 0, xmax_val

    elif dist_escolhida == "Geométrica":
        p_val = col_p1.number_input("p (Sucesso):", min_value=0.0001, max_value=1.0, value=0.30, step=0.05, format="%.4f")
        xmax_val = col_p2.number_input("x_max (Gráfico/Tabela):", min_value=1, value=20, step=1)
        dist = geom(p=p_val)
        nome_dist = f"Geométrica(p={p_val:g})"
        is_discrete = True
        suporte_min, suporte_max = 1, max(1, xmax_val)

    elif dist_escolhida == "Normal":
        mu_val = col_p1.number_input("μ (Média):", value=0.0, step=0.5, format="%.4f")
        sigma_val = col_p2.number_input("σ (Desvio Padrão):", min_value=0.0001, value=1.0, step=0.1, format="%.4f")
        dist = norm(loc=mu_val, scale=sigma_val)
        nome_dist = f"Normal(μ={mu_val:g}, σ={sigma_val:g})"
        suporte_min, suporte_max = mu_val - 4 * sigma_val, mu_val + 4 * sigma_val

    elif dist_escolhida == "Uniforme Contínua":
        a_u = col_p1.number_input("a (Mínimo):", value=0.0, step=0.5, format="%.4f")
        b_u = col_p2.number_input("b (Máximo):", value=10.0, step=0.5, format="%.4f")
        a_val, b_val = min(a_u, b_u), max(a_u, b_u)
        if a_val == b_val:
            b_val += 0.0001
        dist = uniform(loc=a_val, scale=(b_val - a_val))
        nome_dist = f"Uniforme(a={a_val:g}, b={b_val:g})"
        suporte_finito = True
        interv = b_val - a_val
        suporte_min, suporte_max = a_val - 0.2 * interv, b_val + 0.2 * interv

    elif dist_escolhida == "Exponencial":
        taxa_val = col_p1.number_input("λ (Taxa):", min_value=0.0001, value=1.0, step=0.1, format="%.4f")
        dist = expon(scale=1.0 / taxa_val)
        nome_dist = f"Exponencial(λ={taxa_val:g})"
        suporte_min, suporte_max = 0.0, dist.ppf(0.995)

    elif dist_escolhida == "Gama":
        a_g = col_p1.number_input("α (Forma):", min_value=0.0001, value=2.0, step=0.5, format="%.4f")
        b_g = col_p2.number_input("β (Escala):", min_value=0.0001, value=2.0, step=0.2, format="%.4f")
        dist = gamma_dist(a=a_g, scale=b_g)
        nome_dist = f"Gama(α={a_g:g}, β={b_g:g})"
        suporte_min, suporte_max = 0.0, dist.ppf(0.995)

    elif dist_escolhida == "Beta":
        a_b = col_p1.number_input("α (Forma 1):", min_value=0.0001, value=6.0, step=0.5, format="%.4f")
        b_b = col_p2.number_input("β (Forma 2):", min_value=0.0001, value=9.0, step=0.5, format="%.4f")
        dist = beta_dist(a=a_b, b=b_b)
        nome_dist = f"Beta(α={a_b:g}, β={b_b:g})"
        suporte_finito = True
        suporte_min, suporte_max = 0.0, 1.0

    elif dist_escolhida == "Qui-quadrado":
        gl_q = col_p1.number_input("gl (ν):", min_value=0.0001, value=5.0, step=1.0, format="%.4f")
        dist = chi2(df=gl_q)
        nome_dist = f"Qui-quadrado(ν={gl_q:g})"
        suporte_min, suporte_max = 0.0, dist.ppf(0.995)

    elif dist_escolhida == "t-Student":
        gl_t = col_p1.number_input("gl (ν):", min_value=0.0001, value=10.0, step=1.0, format="%.4f")
        dist = t_student(df=gl_t)
        nome_dist = f"t-Student(ν={gl_t:g})"
        suporte_min, suporte_max = dist.ppf(0.001), dist.ppf(0.999)

    elif dist_escolhida == "F":
        d1 = col_p1.number_input("gl1 (d1):", min_value=0.0001, value=5.0, step=1.0, format="%.4f")
        d2 = col_p2.number_input("gl2 (d2):", min_value=0.0001, value=10.0, step=1.0, format="%.4f")
        dist = f_fisher(dfn=d1, dfd=d2)
        nome_dist = f"F(d1={d1:g}, d2={d2:g})"
        suporte_min, suporte_max = 0.0, dist.ppf(0.995)

    elif dist_escolhida == "Cauchy":
        loc_c = col_p1.number_input("x0 (Localização):", value=0.0, step=0.5, format="%.4f")
        sc_c = col_p2.number_input("γ (Escala):", min_value=0.0001, value=1.0, step=0.1, format="%.4f")
        dist = cauchy(loc=loc_c, scale=sc_c)
        nome_dist = f"Cauchy(x0={loc_c:g}, γ={sc_c:g})"
        suporte_min, suporte_max = loc_c - 6 * sc_c, loc_c + 6 * sc_c

    cx1, cx2, ccas, cnot = st.columns([1, 1, 1, 1])
    x1_exp = cx1.number_input("Valor x1:", value=0.25 if dist_escolhida == "Beta" else 3.0, step=0.1, format="%.4f", key="exp_x1")
    x2_exp = cx2.number_input("Valor x2:", value=0.62 if dist_escolhida == "Beta" else 7.0, step=0.1, format="%.4f", key="exp_x2")
    casas_exp = ccas.number_input("Casas decimais:", min_value=1, max_value=12, value=6, step=1, key="exp_casas")
    notacao_exp = cnot.checkbox("Notação científica", value=False)

    a_exp, b_exp = sorted((x1_exp, x2_exp))
    formato_exp = f"{{:.{casas_exp}e}}" if notacao_exp else f"{{:.{casas_exp}f}}"

    if is_discrete:
        prob_igual_x1 = dist.pmf(x1_exp)
        prob_menor_x1 = dist.cdf(x1_exp - 1)
        prob_igual_x2 = dist.pmf(x2_exp)
        prob_menor_x2 = dist.cdf(x2_exp - 1)
        prob_interv_exp = dist.cdf(b_exp) - dist.cdf(a_exp - 1)
        suporte = np.arange(suporte_min, suporte_max + 1)
        valores_prob = dist.pmf(suporte)
        lbl_igual = "P(X = {})"
    else:
        prob_igual_x1 = dist.pdf(x1_exp)
        prob_menor_x1 = dist.cdf(x1_exp)
        prob_igual_x2 = dist.pdf(x2_exp)
        prob_menor_x2 = dist.cdf(x2_exp)
        prob_interv_exp = dist.cdf(b_exp) - dist.cdf(a_exp)
        suporte = np.linspace(suporte_min, suporte_max, 500)
        valores_prob = dist.pdf(suporte)
        lbl_igual = "f({}) [Densidade]"

    media_e, var_e, dp_e = dist.mean(), dist.var(), dist.std()

    col_info, col_graf_exp = st.columns([1, 1.4])
    with col_info:
        st.markdown("##### 📌 Probabilidades & Estatísticas")
        linhas_exp = [
            [lbl_igual.format(f"{x1_exp:g}"), formato_exp.format(prob_igual_x1)],
            [f"P(X < {x1_exp:g})", formato_exp.format(prob_menor_x1)],
            [f"P(X <= {x1_exp:g})", formato_exp.format(dist.cdf(x1_exp))],
            [f"P(X > {x1_exp:g})", formato_exp.format(dist.sf(x1_exp))],
            [lbl_igual.format(f"{x2_exp:g}"), formato_exp.format(prob_igual_x2)],
            [f"P(X < {x2_exp:g})", formato_exp.format(prob_menor_x2)],
            [f"P(X <= {x2_exp:g})", formato_exp.format(dist.cdf(x2_exp))],
            [f"P(X > {x2_exp:g})", formato_exp.format(dist.sf(x2_exp))],
            [f"P({a_exp:g} <= X <= {b_exp:g})", formato_exp.format(prob_interv_exp)],
            ["Média E[X]", formato_exp.format(media_e) if not np.isnan(media_e) else "Indefinida"],
            ["Variância Var[X]", formato_exp.format(var_e) if not np.isnan(var_e) else "Indefinida"],
            ["Desvio Padrão DP[X]", formato_exp.format(dp_e) if not np.isnan(dp_e) else "Indefinida"]
        ]
        df_exp = pd.DataFrame(linhas_exp, columns=["Medida", "Valor"])
        st.table(df_exp.set_index("Medida"))

    with col_graf_exp:
        st.markdown("##### 📈 Gráfico da Distribuição")
        fig_e, ax_e = plt.subplots(figsize=(8.5, 4.5))
        if is_discrete:
            ax_e.bar(suporte, valores_prob, width=0.8, color="#1f77b4", edgecolor="black")
            ax_e.set_ylabel("P(X = x)")
            ax_e.set_title(f"PMF - {nome_dist}")
        else:
            ax_e.plot(suporte, valores_prob, color="#1f77b4", linewidth=2)
            ax_e.fill_between(suporte, valores_prob, alpha=0.1, color="#1f77b4")
            mascara_e = (suporte >= a_exp) & (suporte <= b_exp)
            ax_e.fill_between(
                suporte, valores_prob, where=mascara_e, alpha=0.5, color="#ff7f0e",
                label=f"P({a_exp:g} <= X <= {b_exp:g}) = {formato_exp.format(prob_interv_exp)}"
            )
            ax_e.set_ylabel("Densidade f(x)")
            ax_e.set_title(f"PDF - {nome_dist}")
            ax_e.legend()

        ax_e.axvline(x=x1_exp, linestyle="--", color='red', alpha=0.8)
        ax_e.axvline(x=x2_exp, linestyle="--", color='green', alpha=0.8)
        ax_e.set_xlabel("x")
        ax_e.grid(True, axis="y", linestyle=":")
        fig_e.tight_layout()
        st.pyplot(fig_e)
        plt.close(fig_e)

        if is_discrete:
            with st.expander("🔢 Ver Tabela Completa de Probabilidades (PMF / CDF)"):
                tab_disc = [
                    [int(x), formato_exp.format(dist.pmf(x)), formato_exp.format(dist.cdf(x))]
                    for x in suporte
                ]
                df_tab_disc = pd.DataFrame(tab_disc, columns=["x", "P(X = x)", "P(X <= x) [CDF]"])
                st.dataframe(df_tab_disc, use_container_width=True, hide_index=True)
