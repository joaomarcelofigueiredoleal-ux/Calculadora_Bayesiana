import re
import uuid
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from scipy.stats import (
    binom, poisson, nbinom, norm, expon, gamma as gamma_dist,
    beta as beta_dist, cauchy, uniform, geom, chi2,
    t as t_student, f as f_fisher
)
from scipy.special import gamma as gamma_func, beta as beta_func

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA, FUSO HORÁRIO (BH / UTC-3) E ESTADO DA SESSÃO
# ==============================================================================
st.set_page_config(
    page_title="BayesLab - Estatística Bayesiana e Clássica",
    page_icon="📊",
    layout="wide"
)

FUSO_BR = timezone(timedelta(hours=-3))

if "historico_calculos" not in st.session_state:
    st.session_state["historico_calculos"] = []
if "contador_historico" not in st.session_state:
    st.session_state["contador_historico"] = 0
if "hist_form_ver" not in st.session_state:
    st.session_state["hist_form_ver"] = 0


def renumerar_historico():
    """Garante que os IDs (#) sejam sempre sequenciais (1, 2, 3...) e atualiza o contador."""
    for idx, item in enumerate(st.session_state["historico_calculos"], start=1):
        item["id"] = idx
    st.session_state["contador_historico"] = len(st.session_state["historico_calculos"])


# Migração retroativa para itens já existentes na sessão atual
for _item in st.session_state["historico_calculos"]:
    if "uid" not in _item:
        _item["uid"] = uuid.uuid4().hex
    if "aba_base" not in _item or "rotulo" not in _item:
        _m = re.match(r"^(.*?)\s*\[(.*)\]$", str(_item.get("aba", "")))
        if _m:
            _item["aba_base"] = _m.group(1).strip()
            _item["rotulo"] = _m.group(2).strip()
        else:
            _item["aba_base"] = str(_item.get("aba", "")).strip()
            _item["rotulo"] = ""
    if not _item.get("fuso_corrigido"):
        try:
            _dt = datetime.strptime(_item["hora"], "%H:%M:%S")
            _item["hora"] = (_dt - timedelta(hours=3)).strftime("%H:%M:%S")
        except Exception:
            pass
        _item["fuso_corrigido"] = True

renumerar_historico()


# ==============================================================================
# FUNÇÕES DE QOL: AUTO-DECIMAL, AUTO-SELECT, PARSER DE AMOSTRA E CALLBACKS
# ==============================================================================
def normalizar_para_decimal(val, min_val=0.0):
    """Converte valores > 1 puxando os dígitos para depois da vírgula (ex: 16 -> 0.16)."""
    if val is None:
        return float(min_val)
    val = float(val)
    if val > 1.0:
        digitos_inteiros = len(str(int(abs(val))))
        val = round(val / (10 ** digitos_inteiros), 10)
    return max(float(min_val), min(1.0, val))


def input_decimal_01(container, label, value, key, min_value=0.0, step=0.01, fmt="%.4f"):
    """
    Campo numérico para probabilidades e proporções em [0, 1] sem bloqueio de max_value=1.0
    no frontend, convertendo inteiros digitados (ex: 16 -> 0.16, 05 -> 0.05) automaticamente.
    """
    if key in st.session_state and st.session_state[key] is not None:
        if st.session_state[key] > 1.0:
            st.session_state[key] = normalizar_para_decimal(st.session_state[key], min_value)

    val_bruto = container.number_input(
        label,
        min_value=float(min_value),
        value=float(value),
        step=float(step),
        format=fmt,
        key=key
    )
    return normalizar_para_decimal(val_bruto, min_value)


def parse_amostra_bruta(texto):
    """
    Extrai uma lista de números a partir de texto livre separado por vírgulas,
    ponto-e-vírgula ou espaços (ex: '3, 5, 2, 6' ou '1,5; 2,3; 0,9').
    """
    if not texto or not texto.strip():
        return None
    texto_limpo = texto.strip()
    if ";" in texto_limpo:
        partes = [p.strip().replace(",", ".") for p in texto_limpo.split(";") if p.strip()]
    elif "," in texto_limpo:
        partes = [p.strip() for p in re.split(r"[,\s]+", texto_limpo) if p.strip()]
    else:
        partes = [p.strip().replace(",", ".") for p in texto_limpo.split() if p.strip()]
    try:
        valores = [float(p) for p in partes]
        return valores if len(valores) > 0 else None
    except ValueError:
        return None


def calcular_moda_conjugada(tipo, param1, scale_or_b):
    """Calcula a moda analítica (MAP) para distribuições Beta(a, b) ou Gama(alpha, scale)."""
    if tipo == "beta":
        a, b = param1, scale_or_b
        if a > 1.0 and b > 1.0:
            return (a - 1.0) / (a + b - 2.0)
        elif a == 1.0 and b == 1.0:
            return np.nan
        elif a <= 1.0 and b > 1.0:
            return 0.0
        elif a > 1.0 and b <= 1.0:
            return 1.0
        return np.nan
    elif tipo == "gama":
        alpha, scale = param1, scale_or_b
        return (alpha - 1.0) * scale if alpha >= 1.0 else 0.0
    return np.nan


# Callbacks para Atualização Bayesiana Sequencial (Posteriori -> Nova Priori)
def cb_post_para_priori_discreto(post_vals):
    for idx, val in enumerate(post_vals):
        st.session_state[f"disc_p_{idx}"] = round(float(val), 6)


def cb_post_para_priori_conjugado(key_a, val_a, key_b, val_b):
    st.session_state[key_a] = round(float(val_a), 6)
    st.session_state[key_b] = round(float(val_b), 6)


def atualizar_rotulo_item(item, novo_rotulo):
    """Atualiza o rótulo e reconstrói o nome de exibição da aba no histórico."""
    rot = str(novo_rotulo).strip() if novo_rotulo is not None else ""
    aba_base = item.get("aba_base", item.get("aba", "")).strip()
    item["aba_base"] = aba_base
    item["rotulo"] = rot
    item["aba"] = f"{aba_base} [{rot}]" if rot else aba_base


# Script JS invisível:
# 1) Seleciona automaticamente todo o texto ao focar/clicar em qualquer input numérico
# 2) Puxa inteiros digitados (ex: "27" -> "0.27", "05" -> "0.05") instantaneamente no Tab, Enter ou clique fora
# 3) Abre o campo de renomear rótulo no Histórico ao clicar no botão ✏️ ao lado do nome atual (sem rerun)
components.html(
    """
    <script>
    (function() {
        const win = window.parent;
        const doc = win.document;

        if (win.__bayesLabCleanup) {
            try { win.__bayesLabCleanup(); } catch (e) {}
        }

        function converterSeInteiro(input) {
            if (!input || input.tagName !== 'INPUT') return;
            const step = input.getAttribute('step');
            if (step !== '0.01' && step !== '0.05') return;

            const raw = (input.value || '').trim();
            const match = raw.match(/^(\\d+)(?:[.,]0+)?$/);
            if (!match) return;

            const intPart = match[1];
            if (intPart !== '0' && intPart !== '1') {
                const novoValor = '0.' + intPart;
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    win.HTMLInputElement.prototype, 'value'
                ).set;

                const tracker = input._valueTracker;
                if (tracker) {
                    tracker.setValue(raw);
                }

                nativeSetter.call(input, novoValor);
                input.dispatchEvent(new win.Event('input', { bubbles: true }));
                input.dispatchEvent(new win.Event('change', { bubbles: true }));
            }
        }

        function onFocusIn(e) {
            const el = e.target;
            if (el && el.tagName === 'INPUT' && (el.type === 'number' || el.hasAttribute('step'))) {
                setTimeout(function() {
                    try { el.select(); } catch (err) {}
                }, 15);
            }
        }

        function onKeyDown(e) {
            if (e.key === 'Enter' || e.key === 'Tab') {
                converterSeInteiro(e.target);
            }
        }

        function onFocusOut(e) {
            const el = e.target;
            converterSeInteiro(el);
            setTimeout(function() {
                converterSeInteiro(el);
            }, 10);
        }

        function onClick(e) {
            const btn = e.target.closest ? e.target.closest('.btn-abrir-rename') : null;
            if (!btn) return;
            e.preventDefault();
            e.stopPropagation();

            const viewDiv = btn.closest('.rotulo-view');
            if (!viewDiv) return;
            viewDiv.classList.add('editando');

            const elContainer = viewDiv.closest('.element-container, [data-testid="stElementContainer"]');
            if (elContainer && elContainer.nextElementSibling) {
                setTimeout(function() {
                    const inp = elContainer.nextElementSibling.querySelector('input[type="text"]');
                    if (inp) {
                        inp.focus();
                        inp.select();
                    }
                }, 20);
            }
        }

        win.addEventListener('focusin', onFocusIn, true);
        win.addEventListener('keydown', onKeyDown, true);
        win.addEventListener('focusout', onFocusOut, true);
        win.addEventListener('blur', onFocusOut, true);
        doc.addEventListener('click', onClick, true);

        win.__bayesLabCleanup = function() {
            win.removeEventListener('focusin', onFocusIn, true);
            win.removeEventListener('keydown', onKeyDown, true);
            win.removeEventListener('focusout', onFocusOut, true);
            win.removeEventListener('blur', onFocusOut, true);
            doc.removeEventListener('click', onClick, true);
        };
    })();
    </script>
    """,
    height=0
)


def registrar_historico(aba, resumo_params, resumo_resultado, df_tabela=None, detalhes_extras="", rotulo=""):
    """Salva um cálculo realizado no histórico da sessão no fuso de Brasília (UTC-3)."""
    novo_id = len(st.session_state["historico_calculos"]) + 1
    agora = datetime.now(FUSO_BR).strftime("%H:%M:%S")
    rot_limpo = rotulo.strip() if rotulo else ""
    aba_exibicao = f"{aba} [{rot_limpo}]" if rot_limpo else aba

    item = {
        "uid": uuid.uuid4().hex,
        "id": novo_id,
        "hora": agora,
        "fuso_corrigido": True,
        "aba_base": aba,
        "rotulo": rot_limpo,
        "aba": aba_exibicao,
        "parametros": resumo_params,
        "resultado": resumo_resultado,
        "detalhes": detalhes_extras,
        "tabela": df_tabela.copy() if df_tabela is not None else None
    }
    st.session_state["historico_calculos"].append(item)
    renumerar_historico()
    st.toast(f"✅ Cálculo #{novo_id} ({aba_exibicao}) salvo no Histórico!")


st.title("📊 Suíte de Estatística Bayesiana & Distribuições")

# ==============================================================================
# FUNÇÃO PADRÃO DE RELATÓRIO E GRÁFICO PARA MODELOS CONJUGADOS CONTÍNUOS
# ==============================================================================
def exibir_relatorio_conjugado(dist_priori, dist_post, nome_priori, nome_post,
                               x_vals, x1_val, x2_val, casas_dec, titulo_graf, xlabel_graf,
                               simbolo_param="θ", info_extra="", tipo_dist="beta",
                               params_pri=(1.0, 1.0), params_pos=(1.0, 1.0),
                               modo_intervalo="Limites manuais (x1 e x2)",
                               callback_seq=None, args_seq=None, key_btn_seq=None):
    """Exibe parâmetros, tabela comparativa completa (com MAP, Mediana e ICs) e gráfico."""
    fmt = f"{{:.{casas_dec}f}}"

    if modo_intervalo == "IC 90% da Posteriori":
        a, b = dist_post.ppf(0.05), dist_post.ppf(0.95)
    elif modo_intervalo == "IC 95% da Posteriori":
        a, b = dist_post.ppf(0.025), dist_post.ppf(0.975)
    elif modo_intervalo == "IC 99% da Posteriori":
        a, b = dist_post.ppf(0.005), dist_post.ppf(0.995)
    else:
        a, b = sorted((x1_val, x2_val))

    x1_efetivo, x2_efetivo = a, b

    p_pri_menor_x1 = dist_priori.cdf(x1_efetivo)
    p_pri_maior_x1 = dist_priori.sf(x1_efetivo)
    p_pri_menor_x2 = dist_priori.cdf(x2_efetivo)
    p_pri_maior_x2 = dist_priori.sf(x2_efetivo)
    p_pri_interv   = dist_priori.cdf(b) - dist_priori.cdf(a)

    p_pos_menor_x1 = dist_post.cdf(x1_efetivo)
    p_pos_maior_x1 = dist_post.sf(x1_efetivo)
    p_pos_menor_x2 = dist_post.cdf(x2_efetivo)
    p_pos_maior_x2 = dist_post.sf(x2_efetivo)
    p_pos_interv   = dist_post.cdf(b) - dist_post.cdf(a)

    moda_pri = calcular_moda_conjugada(tipo_dist, params_pri[0], params_pri[1])
    moda_pos = calcular_moda_conjugada(tipo_dist, params_pos[0], params_pos[1])
    str_moda_pri = fmt.format(moda_pri) if not np.isnan(moda_pri) else "Não única / Fronteira"
    str_moda_pos = fmt.format(moda_pos) if not np.isnan(moda_pos) else "Não única / Fronteira"

    ic90_pri = f"[{fmt.format(dist_priori.ppf(0.05))}; {fmt.format(dist_priori.ppf(0.95))}]"
    ic90_pos = f"[{fmt.format(dist_post.ppf(0.05))}; {fmt.format(dist_post.ppf(0.95))}]"
    ic95_pri = f"[{fmt.format(dist_priori.ppf(0.025))}; {fmt.format(dist_priori.ppf(0.975))}]"
    ic95_pos = f"[{fmt.format(dist_post.ppf(0.025))}; {fmt.format(dist_post.ppf(0.975))}]"
    ic99_pri = f"[{fmt.format(dist_priori.ppf(0.005))}; {fmt.format(dist_priori.ppf(0.995))}]"
    ic99_pos = f"[{fmt.format(dist_post.ppf(0.005))}; {fmt.format(dist_post.ppf(0.995))}]"

    c1, c2, c3 = st.columns([1.4, 1.4, 1.1])
    c1.info(f"**Priori (Antes):** `{nome_priori}`")
    c2.error(f"**Posteriori (Depois):** `{nome_post}`")
    if callback_seq is not None:
        c3.button(
            "🔁 Usar Posteriori como Nova Priori",
            on_click=callback_seq,
            args=args_seq,
            key=key_btn_seq,
            use_container_width=True,
            help="Copia os parâmetros da Posteriori atual para os campos da Priori para fazer uma atualização sequencial."
        )
    if info_extra:
        st.caption(f"ℹ️ {info_extra}")

    dados_tabela = [
        [f"P({simbolo_param} < {x1_efetivo:.4g})", fmt.format(p_pri_menor_x1), fmt.format(p_pos_menor_x1)],
        [f"P({simbolo_param} > {x1_efetivo:.4g})", fmt.format(p_pri_maior_x1), fmt.format(p_pos_maior_x1)],
        [f"P({simbolo_param} < {x2_efetivo:.4g})", fmt.format(p_pri_menor_x2), fmt.format(p_pos_menor_x2)],
        [f"P({simbolo_param} > {x2_efetivo:.4g})", fmt.format(p_pri_maior_x2), fmt.format(p_pos_maior_x2)],
        [f"P({a:.4g} < {simbolo_param} < {b:.4g})", fmt.format(p_pri_interv), fmt.format(p_pos_interv)],
        [f"Média E[{simbolo_param}]", fmt.format(dist_priori.mean()), fmt.format(dist_post.mean())],
        [f"Mediana (Quantil 50%)", fmt.format(dist_priori.ppf(0.5)), fmt.format(dist_post.ppf(0.5))],
        [f"Moda / MAP[{simbolo_param}]", str_moda_pri, str_moda_pos],
        [f"Variância Var[{simbolo_param}]", fmt.format(dist_priori.var()), fmt.format(dist_post.var())],
        [f"Desvio Padrão DP[{simbolo_param}]", fmt.format(dist_priori.std()), fmt.format(dist_post.std())],
        ["IC 90% (Simétrico)", ic90_pri, ic90_pos],
        ["IC 95% (Simétrico)", ic95_pri, ic95_pos],
        ["IC 99% (Simétrico)", ic99_pri, ic99_pos]
    ]
    df_res = pd.DataFrame(
        dados_tabela,
        columns=["Medida / Probabilidade", "Priori (Antes)", "Posteriori (Depois)"]
    )

    col_tab, col_graf = st.columns([1.05, 1.35])
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

        ax.fill_between(
            x_vals, priori_pdf, where=mascara, color='#3498db', alpha=0.18,
            label=f'Priori P({a:.4g} < {simbolo_param} < {b:.4g}) = {fmt.format(p_pri_interv)}'
        )
        ax.fill_between(
            x_vals, post_pdf, where=mascara, color='#e74c3c', alpha=0.25,
            label=f'Posteriori P({a:.4g} < {simbolo_param} < {b:.4g}) = {fmt.format(p_pos_interv)}'
        )

        ax.axvline(x=x1_efetivo, linestyle=":", color='darkred', alpha=0.8, label=f'x1 = {x1_efetivo:.4g}')
        ax.axvline(x=x2_efetivo, linestyle=":", color='darkgreen', alpha=0.8, label=f'x2 = {x2_efetivo:.4g}')

        ax.set_title(titulo_graf, fontsize=13)
        ax.set_xlabel(xlabel_graf, fontsize=11)
        ax.set_ylabel('Densidade', fontsize=11)
        ax.legend(loc='best', fontsize=8.8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    resumo_res = (
        f"Posteriori: {nome_post} | "
        f"P({a:.4g} < {simbolo_param} < {b:.4g}) = {fmt.format(p_pos_interv)} | "
        f"E[{simbolo_param}] = {fmt.format(dist_post.mean())} | MAP = {str_moda_pos}"
    )
    return df_res, resumo_res


# ==============================================================================
# ABAS PRINCIPAIS DO APLICATIVO
# ==============================================================================
qtd_hist = len(st.session_state["historico_calculos"])
abas_principais = st.tabs([
    "1. Teorema de Bayes",
    "2. Beta-Binomial",
    "3. Beta-Geométrica",
    "4. Beta-Bin. Negativa",
    "5. Gama-Poisson",
    "6. Gama-Exponencial",
    "7. Explorador de Distribuições",
    f"8. 📜 Histórico ({qtd_hist})"
])

# ==============================================================================
# ABA 1: TEOREMA DE BAYES DISCRETO (COM RAMIFICAÇÕES E AUTOCOMPLETAR)
# ==============================================================================
with abas_principais[0]:
    st.subheader("Teorema de Bayes - Atualização de Probabilidades (Discreto)")

    cfg1, cfg2, cfg3 = st.columns([1.1, 2.2, 1.2])
    num_estados = cfg1.number_input("Nº Opções (Hipóteses):", min_value=2, max_value=10, value=2, step=1)
    modo_veros = cfg2.selectbox(
        "Estrutura da Verossimilhança (Diagrama de Árvore):",
        [
            "Direta — Valor único P(B | Ai)",
            "Sub-ramificações Paralelas — Soma: ∑ P(Sub | Ai) · P(B | Ai, Sub)",
            "Etapas em Série — Produto: ∏ P(Etapa | Ai)"
        ]
    )
    if modo_veros != "Direta — Valor único P(B | Ai)":
        num_subramos = cfg3.number_input(
            "Nº Sub-ramos / Etapas:", min_value=2, max_value=6, value=2, step=1
        )
    else:
        num_subramos = 1

    with st.form("form_bayes_discreto"):
        c2, c3, c4, c5, c6 = st.columns([1.4, 1.4, 1.0, 0.9, 1.2])
        nome_evento = c2.text_input("Evento (Evidência - B):", value="Recessão")
        termo_a = c3.text_input("Termo p/ A (ex: Premissa):", value="Premissa")
        simbolo_a = c4.text_input("Símbolo p/ A (ex: A, H):", value="A")
        casas_disc = c5.number_input("Casas decimais:", min_value=2, max_value=10, value=4, step=1)
        rotulo_disc = c6.text_input("🏷️ Rótulo (opcional):", value="", placeholder="Ex: Questão 1")

        qol_c1, qol_c2 = st.columns(2)
        auto_comp_priori = qol_c1.checkbox(
            "✨ Autocompletar opção deixada em 0,0000 com o complementar (1 - ∑P)",
            value=True,
            help="Se exatamente uma hipótese (ou um sub-ramo) for deixada com probabilidade 0 e as demais somarem menos que 1, preenche automaticamente o valor restante."
        )
        auto_norm_priori = qol_c2.checkbox(
            "⚖️ Normalizar Prioris automaticamente se a soma for diferente de 1",
            value=False
        )

        ev = nome_evento.strip() if nome_evento.strip() else "B"
        t_a = termo_a.strip() if termo_a.strip() else "Opção"
        s_a = simbolo_a.strip() if simbolo_a.strip() else "A"

        st.markdown("##### Defina as Probabilidades a Priori e as Verossimilhanças:")
        estados_raw = []

        for i in range(num_estados):
            padrao_p = 0.80 if i == 0 else (0.20 if i == 1 else 0.0)
            padrao_c = 0.05 if i == 0 else (0.50 if i == 1 else 0.0)

            if modo_veros == "Direta — Valor único P(B | Ai)":
                col_n, col_p, col_v = st.columns([1.5, 1.5, 1.5])
                nome_input = col_n.text_input(
                    f"{t_a} #{i+1}:", value="", placeholder=f"{s_a}{i+1}", key=f"disc_nome_{i}"
                )
                priori_val = input_decimal_01(
                    col_p, f"P({s_a}{i+1}) [Priori]:", value=padrao_p, key=f"disc_p_{i}"
                )
                veros_val = input_decimal_01(
                    col_v, f"P({ev} | {s_a}{i+1}) [Verossim.]:", value=padrao_c, key=f"disc_v_{i}"
                )
                estados_raw.append({
                    "nome_input": nome_input,
                    "priori": priori_val,
                    "veros_direta": veros_val,
                    "subramos": []
                })

            elif modo_veros.startswith("Sub-ramificações Paralelas"):
                st.markdown(f"**{t_a} #{i+1} ({s_a}{i+1})**")
                col_n, col_p = st.columns([1.5, 1.5])
                nome_input = col_n.text_input(
                    f"Nome da {t_a} #{i+1}:", value="", placeholder=f"{s_a}{i+1}", key=f"disc_nome_{i}"
                )
                priori_val = input_decimal_01(
                    col_p, f"P({s_a}{i+1}) [Priori]:", value=padrao_p, key=f"disc_p_{i}"
                )

                subramos_i = []
                for j in range(num_subramos):
                    cs_nome, cs_prob, cs_ver = st.columns([1.4, 1.5, 1.5])
                    padrao_sub_p = round(1.0 / num_subramos, 4) if j < num_subramos - 1 else round(1.0 - round(1.0 / num_subramos, 4) * (num_subramos - 1), 4)
                    padrao_sub_v = min(1.0, round(padrao_c * (0.8 + 0.4 * j), 4))

                    sub_nome = cs_nome.text_input(
                        f"↳ Sub-ramo #{j+1} de {s_a}{i+1}:",
                        value="", placeholder=f"C{i+1}.{j+1}", key=f"sub_n_{i}_{j}"
                    )
                    sub_p = input_decimal_01(
                        cs_prob, f"P(C{i+1}.{j+1} | {s_a}{i+1}):",
                        value=float(padrao_sub_p), key=f"sub_p_{i}_{j}"
                    )
                    sub_v = input_decimal_01(
                        cs_ver, f"P({ev} | {s_a}{i+1} ∩ C{i+1}.{j+1}):",
                        value=float(padrao_sub_v), key=f"sub_v_{i}_{j}"
                    )
                    subramos_i.append({
                        "nome": sub_nome.strip() if sub_nome.strip() else f"C{i+1}.{j+1}",
                        "p_sub": sub_p,
                        "p_ev_sub": sub_v
                    })
                st.markdown("---")
                estados_raw.append({
                    "nome_input": nome_input,
                    "priori": priori_val,
                    "veros_direta": 0.0,
                    "subramos": subramos_i
                })

            else:  # Etapas em Série (Produto)
                st.markdown(f"**{t_a} #{i+1} ({s_a}{i+1})**")
                col_n, col_p = st.columns([1.5, 1.5])
                nome_input = col_n.text_input(
                    f"Nome da {t_a} #{i+1}:", value="", placeholder=f"{s_a}{i+1}", key=f"disc_nome_{i}"
                )
                priori_val = input_decimal_01(
                    col_p, f"P({s_a}{i+1}) [Priori]:", value=padrao_p, key=f"disc_p_{i}"
                )

                cols_etapas = st.columns(num_subramos)
                etapas_i = []
                prod_v = 1.0
                for j in range(num_subramos):
                    padrao_et = round(padrao_c ** (1.0 / num_subramos), 4) if padrao_c > 0 else 0.5
                    val_et = input_decimal_01(
                        cols_etapas[j], f"↳ Etapa #{j+1} P(E{j+1} | {s_a}{i+1}):",
                        value=float(padrao_et), key=f"et_v_{i}_{j}"
                    )
                    prod_v *= val_et
                    etapas_i.append({
                        "nome": f"E{j+1}",
                        "p_etapa": val_et
                    })
                st.markdown("---")
                estados_raw.append({
                    "nome_input": nome_input,
                    "priori": priori_val,
                    "veros_direta": prod_v,
                    "subramos": etapas_i
                })

        sub_bayes = st.form_submit_button("🔄 Calcular / Atualizar", type="primary")

    fmt_disc = f"{{:.{casas_disc}f}}"
    fmt_pct = f"{{:.{max(2, casas_disc - 2)}f}}%"

    avisos_autocompletar = []
    if auto_comp_priori:
        zeros_idx = [idx for idx, it in enumerate(estados_raw) if np.isclose(it["priori"], 0.0)]
        soma_atual = sum(it["priori"] for it in estados_raw)
        if len(zeros_idx) == 1 and 0.0 < soma_atual < 1.0:
            idx_z = zeros_idx[0]
            comp_val = round(1.0 - soma_atual, 10)
            estados_raw[idx_z]["priori"] = comp_val
            nome_z = estados_raw[idx_z]["nome_input"].strip() or f"{s_a}{idx_z+1}"
            avisos_autocompletar.append(f"Priori de **{nome_z}** autocompletada com **{fmt_disc.format(comp_val)}**.")

    if modo_veros.startswith("Sub-ramificações Paralelas"):
        for idx, it in enumerate(estados_raw):
            subs = it["subramos"]
            if auto_comp_priori:
                z_sub = [j for j, s in enumerate(subs) if np.isclose(s["p_sub"], 0.0)]
                soma_s = sum(s["p_sub"] for s in subs)
                if len(z_sub) == 1 and 0.0 < soma_s < 1.0:
                    j_z = z_sub[0]
                    comp_s = round(1.0 - soma_s, 10)
                    subs[j_z]["p_sub"] = comp_s
                    avisos_autocompletar.append(
                        f"Sub-ramo **{subs[j_z]['nome']}** autocompletado com **{fmt_disc.format(comp_s)}**."
                    )
            it["veros_direta"] = sum(s["p_sub"] * s["p_ev_sub"] for s in subs)

    soma_bruta = sum(it["priori"] for it in estados_raw)
    if auto_norm_priori and soma_bruta > 0 and not np.isclose(soma_bruta, 1.0):
        for it in estados_raw:
            it["priori"] = it["priori"] / soma_bruta
        avisos_autocompletar.append(f"Prioris normalizadas automaticamente (soma original era {soma_bruta:.4g}).")

    estados_dict = {}
    for i, item in enumerate(estados_raw):
        nome_val = item["nome_input"].strip() if item["nome_input"].strip() else f"{s_a}{i+1}"
        if nome_val in estados_dict:
            nome_val = f"{nome_val} ({i+1})"
        estados_dict[nome_val] = {
            "priori": item["priori"],
            "veros": item["veros_direta"],
            "subramos": item["subramos"]
        }

    nomes_lista = list(estados_dict.keys())
    soma_priors = sum(estados_dict[k]["priori"] for k in nomes_lista)
    p_ev = sum(estados_dict[k]["priori"] * estados_dict[k]["veros"] for k in nomes_lista)

    st.divider()
    if avisos_autocompletar:
        st.caption("✨ " + " | ".join(avisos_autocompletar))

    col_sel, col_res1, col_res2, col_seq = st.columns([1.1, 1.3, 1.3, 1.1])
    alvo = col_sel.selectbox("🎯 Destacar Posteriori de:", options=nomes_lista)

    if not np.isclose(soma_priors, 1.0, atol=0.01):
        st.warning(f"⚠️ Atenção: A soma das probabilidades a priori é **{fmt_disc.format(soma_priors)}** (deveria ser 1.0).")

    if modo_veros.startswith("Sub-ramificações Paralelas"):
        for nome in nomes_lista:
            soma_sub = sum(s["p_sub"] for s in estados_dict[nome]["subramos"])
            if not np.isclose(soma_sub, 1.0, atol=0.01):
                st.warning(
                    f"⚠️ Atenção: A soma das probabilidades dos sub-ramos de **{nome}** é "
                    f"**{fmt_disc.format(soma_sub)}** (caso sejam exaustivos, deveria ser 1.0)."
                )

    if p_ev == 0:
        st.error(f"❌ Erro: A probabilidade total do evento P({ev}) é 0.")
    else:
        priori_alvo = estados_dict[alvo]["priori"]
        veros_alvo = estados_dict[alvo]["veros"]
        posteriori_alvo = (priori_alvo * veros_alvo) / p_ev

        tabela_dados = []
        priors_lista = []
        posteriors_lista = []

        for nome in nomes_lista:
            pri = estados_dict[nome]["priori"]
            ver = estados_dict[nome]["veros"]
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

        col_res1.metric(
            label=f"Probabilidade Total P({ev})",
            value=f"{fmt_disc.format(p_ev)} ({fmt_pct.format(p_ev*100)})"
        )
        col_res2.metric(
            label=f"Posteriori em Destaque P({alvo} | {ev})",
            value=f"{fmt_disc.format(posteriori_alvo)} ({fmt_pct.format(posteriori_alvo*100)})"
        )
        col_seq.write("")
        col_seq.button(
            "🔁 Usar Posteriori como Nova Priori",
            on_click=cb_post_para_priori_discreto,
            args=(posteriors_lista,),
            key="btn_seq_disc",
            use_container_width=True,
            help="Substitui as probabilidades a Priori pelas probabilidades a Posteriori atuais para uma nova rodada."
        )

        sub_disc = st.tabs(["📋 Tabela de Bayes", "📊 Gráfico de Barras", "🌳 Diagrama de Árvore"])

        df_disc = pd.DataFrame(
            tabela_dados,
            columns=[t_a, f"P({t_a}) [Priori]", f"P({ev} | {t_a}) [Verossim.]", "Conjunta", f"P({t_a} | {ev}) [Posteriori]"]
        )

        with sub_disc[0]:
            st.table(df_disc.set_index(t_a))

            if modo_veros.startswith("Sub-ramificações Paralelas"):
                st.markdown("##### 🌿 Detalhamento por Sub-ramificação")
                linhas_sub = []
                for nome in nomes_lista:
                    pri = estados_dict[nome]["priori"]
                    for s in estados_dict[nome]["subramos"]:
                        conj_sub = pri * s["p_sub"] * s["p_ev_sub"]
                        post_sub = conj_sub / p_ev
                        linhas_sub.append([
                            nome,
                            s["nome"],
                            fmt_disc.format(pri),
                            fmt_disc.format(s["p_sub"]),
                            fmt_disc.format(s["p_ev_sub"]),
                            fmt_disc.format(s["p_sub"] * s["p_ev_sub"]),
                            fmt_disc.format(conj_sub),
                            f"{fmt_disc.format(post_sub)} ({fmt_pct.format(post_sub*100)})"
                        ])
                df_sub = pd.DataFrame(
                    linhas_sub,
                    columns=[
                        t_a, "Sub-ramo (C)", f"P({t_a})", f"P(C | {t_a})",
                        f"P({ev} | {t_a} ∩ C)", f"Parcela em P({ev} | {t_a})",
                        f"Conjunta P({t_a} ∩ C ∩ {ev})", f"Posteriori P({t_a} ∩ C | {ev})"
                    ]
                )
                st.dataframe(df_sub, use_container_width=True, hide_index=True)

            elif modo_veros.startswith("Etapas em Série"):
                st.markdown("##### 🔗 Detalhamento das Etapas em Série na Verossimilhança")
                linhas_et = []
                for nome in nomes_lista:
                    etapas_str = " × ".join(f"{s['p_etapa']:.4g}" for s in estados_dict[nome]["subramos"])
                    linhas_et.append([
                        nome,
                        etapas_str,
                        fmt_disc.format(estados_dict[nome]["veros"])
                    ])
                df_et = pd.DataFrame(
                    linhas_et,
                    columns=[t_a, "Produto das Etapas", f"P({ev} | {t_a}) Resultante"]
                )
                st.table(df_et.set_index(t_a))

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
            if modo_veros.startswith("Sub-ramificações Paralelas"):
                total_sub_ativos = sum(len(estados_dict[n]["subramos"]) for n in nomes_lista)
                fig_tree, ax_tree = plt.subplots(figsize=(13.5, max(5.5, total_sub_ativos * 1.45)))
                ax_tree.axis('off')
                ax_tree.text(0, 0.5, 'Início', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray'))

                y_step_c = 1.0 / (total_sub_ativos + 1)
                idx_c_global = 0

                for nome in nomes_lista:
                    pri = estados_dict[nome]["priori"]
                    subs = estados_dict[nome]["subramos"]
                    ys_deste_a = []

                    for s in subs:
                        idx_c_global += 1
                        y_c = 1.0 - idx_c_global * y_step_c
                        ys_deste_a.append((y_c, s))

                    y_a = np.mean([par[0] for par in ys_deste_a])

                    ax_tree.plot([0.08, 0.88], [0.5, y_a], 'k-', lw=1.6)
                    ax_tree.text(
                        0.48, (0.5 + y_a)/2 + 0.015, f"P({nome})={pri:.4g}",
                        ha='center', va='bottom', fontsize=9, color='darkblue'
                    )
                    ax_tree.text(
                        1.0, y_a, nome, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.45', fc='lightblue', ec='blue')
                    )

                    for y_c, s in ys_deste_a:
                        p_sub = s["p_sub"]
                        p_ev_sub = s["p_ev_sub"]
                        p_nao_ev_sub = 1.0 - p_ev_sub
                        nome_c = s["nome"]

                        ax_tree.plot([1.12, 2.05], [y_a, y_c], 'k-', lw=1.3)
                        ax_tree.text(
                            1.58, (y_a + y_c)/2 + 0.012, f"P({nome_c}|{nome})={p_sub:.4g}",
                            ha='center', va='bottom', fontsize=8.5, color='#5b2c6f'
                        )
                        ax_tree.text(
                            2.2, y_c, nome_c, ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.38', fc='#ebdef0', ec='#8e44ad')
                        )

                        y_up = y_c + y_step_c / 3.3
                        y_down = y_c - y_step_c / 3.3

                        ax_tree.plot([2.34, 3.35], [y_c, y_up], 'k-', lw=1.3)
                        ax_tree.text(
                            2.85, (y_c + y_up)/2 + 0.01, f"P({ev}|{nome},{nome_c})={p_ev_sub:.4g}",
                            ha='center', va='bottom', fontsize=8, color='green'
                        )
                        ax_tree.text(
                            3.5, y_up, ev, ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.32', fc='lightgreen', ec='green')
                        )
                        ax_tree.text(
                            3.72, y_up, f"∩ = {fmt_disc.format(pri * p_sub * p_ev_sub)}",
                            ha='left', va='center', fontsize=8.8, fontweight='bold'
                        )

                        ax_tree.plot([2.34, 3.35], [y_c, y_down], 'k--', lw=1.1, alpha=0.55)
                        ax_tree.text(
                            2.85, (y_c + y_down)/2 - 0.01, f"1-P={p_nao_ev_sub:.4g}",
                            ha='center', va='top', fontsize=8, color='red'
                        )
                        ax_tree.text(
                            3.5, y_down, f"Não {ev}", ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.32', fc='#ffcccc', ec='red')
                        )
                        ax_tree.text(
                            3.72, y_down, f"∩ = {fmt_disc.format(pri * p_sub * p_nao_ev_sub)}",
                            ha='left', va='center', fontsize=8.5, alpha=0.6
                        )

                ax_tree.set_xlim(-0.2, 4.55)
                ax_tree.set_ylim(0, 1)
                ax_tree.set_title(f'Diagrama de Árvore com Sub-ramificações: P({ev})', fontsize=13, pad=15)
                fig_tree.tight_layout()
                st.pyplot(fig_tree)
                plt.close(fig_tree)

            else:
                fig_tree, ax_tree = plt.subplots(figsize=(11.5, max(4.5, num_estados * 1.65)))
                ax_tree.axis('off')
                ax_tree.text(0, 0.5, 'Início', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray'))

                y_step = 1.0 / (num_estados + 1)
                for i, nome in enumerate(nomes_lista):
                    y_a = 1.0 - (i + 1) * y_step
                    pri = estados_dict[nome]["priori"]
                    ver = estados_dict[nome]["veros"]
                    prob_nao_ev = 1.0 - ver

                    ax_tree.plot([0.1, 0.9], [0.5, y_a], 'k-', lw=1.5)
                    ax_tree.text(0.5, (0.5 + y_a)/2 + 0.02, f"P({nome})={pri:.4g}", ha='center', va='bottom', fontsize=9.5, color='darkblue')
                    ax_tree.text(1.0, y_a, nome, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', ec='blue'))

                    y_up = y_a + y_step / 3.5
                    y_down = y_a - y_step / 3.5

                    if modo_veros.startswith("Etapas em Série"):
                        prod_txt = "×".join(f"{s['p_etapa']:.3g}" for s in estados_dict[nome]["subramos"])
                        lbl_ver = f"P({ev}|{nome}) = {prod_txt} = {ver:.4g}"
                    else:
                        lbl_ver = f"P({ev}|{nome})={ver:.4g}"

                    ax_tree.plot([1.1, 2.3], [y_a, y_up], 'k-', lw=1.5)
                    ax_tree.text(1.7, (y_a + y_up)/2 + 0.02, lbl_ver, ha='center', va='bottom', fontsize=8.8, color='green')
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

        if sub_bayes:
            resumo_p = (
                f"Evento: {ev} | Modo: {modo_veros.split('—')[0].strip()} | "
                + ", ".join(f"{n}: Pri={estados_dict[n]['priori']:.4g}, Ver={estados_dict[n]['veros']:.4g}" for n in nomes_lista)
            )
            resumo_r = (
                f"P({ev}) = {fmt_disc.format(p_ev)} | "
                + ", ".join(
                    f"P({n}|{ev}) = {fmt_disc.format((estados_dict[n]['priori']*estados_dict[n]['veros'])/p_ev)}"
                    for n in nomes_lista
                )
            )
            registrar_historico("1. Teorema de Bayes", resumo_p, resumo_r, df_disc, rotulo=rotulo_disc)


# ==============================================================================
# ABA 2: MODELO BETA-BINOMIAL
# ==============================================================================
with abas_principais[1]:
    st.subheader("Modelo Beta-Binomial (Proporções em 'n' ensaios fixos)")
    with st.form("form_bb"):
        c1, c2, c3, c4 = st.columns(4)
        bb_a = c1.number_input("Priori 'a':", min_value=0.0001, value=4.0, step=0.5, key="bb_a")
        bb_b = c2.number_input("Priori 'b':", min_value=0.0001, value=2.0, step=0.5, key="bb_b")
        bb_n = c3.number_input("Ensaios (n):", min_value=1, value=4, step=1, key="bb_n")
        bb_k = c4.number_input("Sucessos (k):", min_value=0, value=3, step=1, key="bb_k")

        c5, c6, c7, c8, c9 = st.columns([1, 1, 1.3, 0.8, 1.1])
        bb_x1 = input_decimal_01(c5, "Limite x1:", value=0.50, step=0.05, key="bb_x1")
        bb_x2 = input_decimal_01(c6, "Limite x2:", value=0.80, step=0.05, key="bb_x2")
        bb_modo_ic = c7.selectbox(
            "Intervalo sombreado:",
            ["Limites manuais (x1 e x2)", "IC 90% da Posteriori", "IC 95% da Posteriori", "IC 99% da Posteriori"],
            key="bb_modo_ic"
        )
        bb_casas = c8.number_input("Casas dec.:", min_value=1, max_value=10, value=5, step=1, key="bb_casas")
        bb_rotulo = c9.text_input("🏷️ Rótulo:", value="", placeholder="Ex: Q2a", key="bb_rot")

        sub_bb = st.form_submit_button("🔄 Calcular / Atualizar", type="primary")

    if bb_k > bb_n:
        st.error("❌ Erro: O número de sucessos (k) não pode ser maior que o de ensaios (n).")
    else:
        a_post = bb_a + bb_k
        b_post = bb_b + bb_n - bb_k
        d_pri = beta_dist(bb_a, bb_b)
        d_pos = beta_dist(a_post, b_post)
        theta = np.linspace(0, 1, 500)
        int_norm = beta_func(a_post, b_post) / beta_func(bb_a, bb_b)
        prob_pred_prox = a_post / (a_post + b_post)
        info = (
            f"Sucessos: {bb_k} | Fracassos: {bb_n - bb_k} | "
            f"Integral do Núcleo: {int_norm:.6e} | "
            f"Prob. Preditiva de Sucesso no Próximo Ensaio: {prob_pred_prox:.{bb_casas}f}"
        )

        df_bb, res_bb = exibir_relatorio_conjugado(
            d_pri, d_pos, f"Beta({bb_a:g}; {bb_b:g})", f"Beta({a_post:g}; {b_post:g})",
            theta, bb_x1, bb_x2, bb_casas,
            "Atualização Bayesiana (Modelo Beta-Binomial)", r"$\theta$ (Probabilidade de Sucesso)", "θ", info,
            tipo_dist="beta", params_pri=(bb_a, bb_b), params_pos=(a_post, b_post),
            modo_intervalo=bb_modo_ic,
            callback_seq=cb_post_para_priori_conjugado, args_seq=("bb_a", a_post, "bb_b", b_post), key_btn_seq="btn_seq_bb"
        )
        if sub_bb:
            registrar_historico(
                "2. Beta-Binomial",
                f"Priori: Beta({bb_a:g}; {bb_b:g}) | n={bb_n}, k={bb_k} | Intervalo: [{bb_x1:g}, {bb_x2:g}]",
                res_bb, df_bb, info, rotulo=bb_rotulo
            )


# ==============================================================================
# ABA 3: MODELO BETA-GEOMÉTRICA
# ==============================================================================
with abas_principais[2]:
    st.subheader("Modelo Beta-Geométrica (Ensaios até o 1º sucesso por sequência)")
    with st.form("form_bg"):
        c1, c2, c3, c4 = st.columns(4)
        bg_a = c1.number_input("Priori 'a':", min_value=0.0001, value=6.0, step=0.5, key="bg_a")
        bg_b = c2.number_input("Priori 'b':", min_value=0.0001, value=9.0, step=0.5, key="bg_b")
        bg_n_in = c3.number_input("Total de Ensaios (n = ∑x):", min_value=1, value=28, step=1, key="bg_n")
        bg_m_in = c4.number_input("Sequências / Sucessos (m):", min_value=1, value=12, step=1, key="bg_m")

        bg_raw = st.text_input(
            "📥 Opcional — Colar amostra de ensaios por sequência (ex: 2, 1, 4, 3 — substitui 'n' e 'm' automaticamente):",
            value="", key="bg_raw"
        )

        c5, c6, c7, c8, c9 = st.columns([1, 1, 1.3, 0.8, 1.1])
        bg_x1 = input_decimal_01(c5, "Limite x1:", value=0.25, step=0.05, key="bg_x1")
        bg_x2 = input_decimal_01(c6, "Limite x2:", value=0.62, step=0.05, key="bg_x2")
        bg_modo_ic = c7.selectbox(
            "Intervalo sombreado:",
            ["Limites manuais (x1 e x2)", "IC 90% da Posteriori", "IC 95% da Posteriori", "IC 99% da Posteriori"],
            key="bg_modo_ic"
        )
        bg_casas = c8.number_input("Casas dec.:", min_value=1, max_value=10, value=5, step=1, key="bg_casas")
        bg_rotulo = c9.text_input("🏷️ Rótulo:", value="", placeholder="Ex: Q3", key="bg_rot")

        sub_bg = st.form_submit_button("🔄 Calcular / Atualizar", type="primary")

    amostra_bg = parse_amostra_bruta(bg_raw)
    if amostra_bg is not None:
        bg_m = len(amostra_bg)
        bg_n = int(round(sum(amostra_bg)))
        st.caption(f"📥 Amostra lida: **{bg_m} sequências (m)** | **Soma de ensaios (n) = {bg_n}** | Média = {np.mean(amostra_bg):.4g}")
    else:
        bg_n, bg_m = bg_n_in, bg_m_in

    if bg_m > bg_n:
        st.error("❌ Erro: O número de sequências (m) não pode ser maior que o total de ensaios (n).")
    else:
        fracassos = bg_n - bg_m
        a_post = bg_a + bg_m
        b_post = bg_b + fracassos
        d_pri = beta_dist(bg_a, bg_b)
        d_pos = beta_dist(a_post, b_post)
        theta = np.linspace(0, 1, 500)
        info = f"Sucessos (sequências m): {bg_m} | Fracassos (n - m): {fracassos}"

        df_bg, res_bg = exibir_relatorio_conjugado(
            d_pri, d_pos, f"Beta({bg_a:g}; {bg_b:g})", f"Beta({a_post:g}; {b_post:g})",
            theta, bg_x1, bg_x2, bg_casas,
            "Atualização Bayesiana (Modelo Beta-Geométrica)", r"$\theta$ (Probabilidade de Sucesso)", "θ", info,
            tipo_dist="beta", params_pri=(bg_a, bg_b), params_pos=(a_post, b_post),
            modo_intervalo=bg_modo_ic,
            callback_seq=cb_post_para_priori_conjugado, args_seq=("bg_a", a_post, "bg_b", b_post), key_btn_seq="btn_seq_bg"
        )
        if sub_bg:
            registrar_historico(
                "3. Beta-Geométrica",
                f"Priori: Beta({bg_a:g}; {bg_b:g}) | n={bg_n}, m={bg_m} | Intervalo: [{bg_x1:g}, {bg_x2:g}]",
                res_bg, df_bg, info, rotulo=bg_rotulo
            )


# ==============================================================================
# ABA 4: MODELO BETA-BINOMIAL NEGATIVA
# ==============================================================================
with abas_principais[3]:
    st.subheader("Modelo Beta-Binomial Negativa (Ensaios até atingir 'r' sucessos)")
    with st.form("form_bbn"):
        c1, c2, c3, c4 = st.columns(4)
        bbn_a = c1.number_input("Priori 'a':", min_value=0.0001, value=12.0, step=0.5, key="bbn_a")
        bbn_b = c2.number_input("Priori 'b':", min_value=0.0001, value=7.0, step=0.5, key="bbn_b")
        bbn_n = c3.number_input("Total de Ensaios (n):", min_value=1, value=24, step=1, key="bbn_n")
        bbn_r = c4.number_input("Sucessos fixos (r):", min_value=1, value=9, step=1, key="bbn_r")

        c5, c6, c7, c8, c9 = st.columns([1, 1, 1.3, 0.8, 1.1])
        bbn_x1 = input_decimal_01(c5, "Limite x1:", value=0.30, step=0.05, key="bbn_x1")
        bbn_x2 = input_decimal_01(c6, "Limite x2:", value=0.75, step=0.05, key="bbn_x2")
        bbn_modo_ic = c7.selectbox(
            "Intervalo sombreado:",
            ["Limites manuais (x1 e x2)", "IC 90% da Posteriori", "IC 95% da Posteriori", "IC 99% da Posteriori"],
            key="bbn_modo_ic"
        )
        bbn_casas = c8.number_input("Casas dec.:", min_value=1, max_value=10, value=5, step=1, key="bbn_casas")
        bbn_rotulo = c9.text_input("🏷️ Rótulo:", value="", placeholder="Ex: Q4", key="bbn_rot")

        sub_bbn = st.form_submit_button("🔄 Calcular / Atualizar", type="primary")

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

        df_bbn, res_bbn = exibir_relatorio_conjugado(
            d_pri, d_pos, f"Beta({bbn_a:g}; {bbn_b:g})", f"Beta({a_post:g}; {b_post:g})",
            theta, bbn_x1, bbn_x2, bbn_casas,
            "Atualização Bayesiana (Modelo Beta-Binomial Negativa)", r"$\theta$ (Probabilidade de Sucesso)", "θ", info,
            tipo_dist="beta", params_pri=(bbn_a, bbn_b), params_pos=(a_post, b_post),
            modo_intervalo=bbn_modo_ic,
            callback_seq=cb_post_para_priori_conjugado, args_seq=("bbn_a", a_post, "bbn_b", b_post), key_btn_seq="btn_seq_bbn"
        )
        if sub_bbn:
            registrar_historico(
                "4. Beta-Bin. Negativa",
                f"Priori: Beta({bbn_a:g}; {bbn_b:g}) | n={bbn_n}, r={bbn_r} | Intervalo: [{bbn_x1:g}, {bbn_x2:g}]",
                res_bbn, df_bbn, info, rotulo=bbn_rotulo
            )


# ==============================================================================
# ABA 5: MODELO GAMA-POISSON
# ==============================================================================
with abas_principais[4]:
    st.subheader("Modelo Gama-Poisson (Taxas de Contagem)")
    with st.form("form_gp"):
        c0, c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1, 1])
        gp_param = c0.selectbox("Parametrização:", ["Escala (β)", "Taxa (β)"], key="gp_param")
        gp_alpha = c1.number_input("Priori alfa (α):", min_value=0.0001, value=3.0, step=0.5, format="%.4f", key="gp_a")
        gp_beta = c2.number_input("Priori beta (β):", min_value=0.0001, value=1.0, step=0.1, format="%.4f", key="gp_b")
        gp_n_in = c3.number_input("Períodos (n):", min_value=1, value=4, step=1, key="gp_n")
        gp_soma_in = c4.number_input("Soma (Σx):", min_value=0, value=10, step=1, key="gp_soma")

        gp_raw = st.text_input(
            "📥 Opcional — Colar contagens observadas (ex: 2, 4, 1, 3 — calcula 'n' e 'Σx' automaticamente):",
            value="", key="gp_raw"
        )

        c5, c6, c7, c8, c9 = st.columns([1, 1, 1.3, 0.8, 1.1])
        gp_x1 = c5.number_input("Limite x1:", value=1.50, step=0.1, format="%.4f", key="gp_x1")
        gp_x2 = c6.number_input("Limite x2:", value=3.70, step=0.1, format="%.4f", key="gp_x2")
        gp_modo_ic = c7.selectbox(
            "Intervalo sombreado:",
            ["Limites manuais (x1 e x2)", "IC 90% da Posteriori", "IC 95% da Posteriori", "IC 99% da Posteriori"],
            key="gp_modo_ic"
        )
        gp_casas = c8.number_input("Casas dec.:", min_value=1, max_value=10, value=5, step=1, key="gp_casas")
        gp_rotulo = c9.text_input("🏷️ Rótulo:", value="", placeholder="Ex: Q5", key="gp_rot")

        sub_gp = st.form_submit_button("🔄 Calcular / Atualizar", type="primary")

    amostra_gp = parse_amostra_bruta(gp_raw)
    if amostra_gp is not None:
        gp_n = len(amostra_gp)
        gp_soma = int(round(sum(amostra_gp)))
        st.caption(f"📥 Amostra lida: **n = {gp_n}** | **Σx = {gp_soma}** | Média amostral (x̄) = {np.mean(amostra_gp):.4g}")
    else:
        gp_n, gp_soma = gp_n_in, gp_soma_in

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
    info = f"Parametrização por {rotulo_tipo} | Média amostral x̄ = {gp_soma/gp_n:.4g} | Integral de Normalização (núcleo): {int_nucleo:.6f}"
    if gp_n == 1:
        int_completa = int_nucleo / gamma_func(gp_soma + 1)
        info += f" | Com 1/x!: {int_completa:.6f}"

    limite_sup = max(d_pri.ppf(0.995), d_pos.ppf(0.995), gp_x1 * 1.2, gp_x2 * 1.2, 0.1)
    lambda_vals = np.linspace(0, limite_sup, 500)

    df_gp, res_gp = exibir_relatorio_conjugado(
        d_pri, d_pos, f"Gama({gp_alpha:g}; {gp_beta:g})", f"Gama({a_post:g}; {b_post:.6f})",
        lambda_vals, gp_x1, gp_x2, gp_casas,
        "Atualização Bayesiana (Modelo Gama-Poisson)", r"$\lambda$ (Taxa de Ocorrência)", "λ", info,
        tipo_dist="gama", params_pri=(gp_alpha, scale_pri), params_pos=(a_post, scale_pos),
        modo_intervalo=gp_modo_ic,
        callback_seq=cb_post_para_priori_conjugado, args_seq=("gp_a", a_post, "gp_b", b_post), key_btn_seq="btn_seq_gp"
    )
    if sub_gp:
        registrar_historico(
            "5. Gama-Poisson",
            f"Priori: Gama({gp_alpha:g}; {gp_beta:g}) [{gp_param}] | n={gp_n}, Σx={gp_soma} | Intervalo: [{gp_x1:g}, {gp_x2:g}]",
            res_gp, df_gp, info, rotulo=gp_rotulo
        )


# ==============================================================================
# ABA 6: MODELO GAMA-EXPONENCIAL
# ==============================================================================
with abas_principais[5]:
    st.subheader("Modelo Gama-Exponencial (Tempos até Evento)")
    with st.form("form_ge"):
        c0, c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1, 1])
        ge_param = c0.selectbox("Parametrização:", ["Escala (β)", "Taxa (β)"], key="ge_param")
        ge_alpha = c1.number_input("Priori alfa (α):", min_value=0.0001, value=3.0, step=0.5, format="%.4f", key="ge_a")
        ge_beta = c2.number_input("Priori beta (β):", min_value=0.0001, value=1.0, step=0.1, format="%.4f", key="ge_b")
        ge_n_in = c3.number_input("Observações (n):", min_value=1, value=5, step=1, key="ge_n")
        ge_soma_in = c4.number_input("Soma dos tempos (Σx):", min_value=0.0001, value=5.0, step=0.5, format="%.4f", key="ge_soma")

        ge_raw = st.text_input(
            "📥 Opcional — Colar tempos observados (ex: 0.8, 1.2, 0.5, 1.5, 1.0 — calcula 'n' e 'Σx' automaticamente):",
            value="", key="ge_raw"
        )

        c5, c6, c7, c8, c9 = st.columns([1, 1, 1.3, 0.8, 1.1])
        ge_x1 = c5.number_input("Limite x1:", value=0.70, step=0.1, format="%.4f", key="ge_x1")
        ge_x2 = c6.number_input("Limite x2:", value=1.60, step=0.1, format="%.4f", key="ge_x2")
        ge_modo_ic = c7.selectbox(
            "Intervalo sombreado:",
            ["Limites manuais (x1 e x2)", "IC 90% da Posteriori", "IC 95% da Posteriori", "IC 99% da Posteriori"],
            key="ge_modo_ic"
        )
        ge_casas = c8.number_input("Casas dec.:", min_value=1, max_value=10, value=5, step=1, key="ge_casas")
        ge_rotulo = c9.text_input("🏷️ Rótulo:", value="", placeholder="Ex: Q6", key="ge_rot")

        sub_ge = st.form_submit_button("🔄 Calcular / Atualizar", type="primary")

    amostra_ge = parse_amostra_bruta(ge_raw)
    if amostra_ge is not None:
        ge_n = len(amostra_ge)
        ge_soma = float(sum(amostra_ge))
        st.caption(f"📥 Amostra lida: **n = {ge_n}** | **Σx = {ge_soma:.4g}** | Tempo médio (x̄) = {np.mean(amostra_ge):.4g}")
    else:
        ge_n, ge_soma = ge_n_in, ge_soma_in

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
    info = f"Parametrização por {rotulo_tipo} | Tempo médio amostral x̄ = {ge_soma/ge_n:.4g} | Integral de Normalização: {int_norm:.6f}"

    limite_sup = max(d_pri.ppf(0.995), d_pos.ppf(0.995), ge_x1 * 1.2, ge_x2 * 1.2, 0.1)
    lambda_vals = np.linspace(0, limite_sup, 500)

    df_ge, res_ge = exibir_relatorio_conjugado(
        d_pri, d_pos, f"Gama({ge_alpha:g}; {ge_beta:g})", f"Gama({a_post:g}; {b_post:.6f})",
        lambda_vals, ge_x1, ge_x2, ge_casas,
        "Atualização Bayesiana (Modelo Gama-Exponencial)", r"$\lambda$ (Taxa de Falha / Tempo)", "λ", info,
        tipo_dist="gama", params_pri=(ge_alpha, scale_pri), params_pos=(a_post, scale_pos),
        modo_intervalo=ge_modo_ic,
        callback_seq=cb_post_para_priori_conjugado, args_seq=("ge_a", a_post, "ge_b", b_post), key_btn_seq="btn_seq_ge"
    )
    if sub_ge:
        registrar_historico(
            "6. Gama-Exponencial",
            f"Priori: Gama({ge_alpha:g}; {ge_beta:g}) [{ge_param}] | n={ge_n}, Σx={ge_soma:g} | Intervalo: [{ge_x1:g}, {ge_x2:g}]",
            res_ge, df_ge, info, rotulo=ge_rotulo
        )


# ==============================================================================
# ABA 7: EXPLORADOR DE DISTRIBUIÇÕES (COM CALCULADORA INVERSA DE QUANTIS)
# ==============================================================================
with abas_principais[6]:
    st.subheader("Explorador de Distribuições de Probabilidade")

    dist_escolhida = st.selectbox(
        "Selecione a Distribuição:",
        [
            "Binomial", "Poisson", "Binomial Negativa", "Geométrica",
            "Normal", "Uniforme Contínua", "Exponencial", "Gama",
            "Beta", "Qui-quadrado", "t-Student", "F", "Cauchy"
        ]
    )

    is_discrete = False
    suporte_finito = False

    with st.form("form_explorador"):
        col_p1, col_p2 = st.columns(2)

        if dist_escolhida == "Binomial":
            n_val = col_p1.number_input("n (Ensaios):", min_value=1, value=10, step=1)
            p_val = input_decimal_01(col_p2, "p (Sucesso):", value=0.30, step=0.05, key="exp_binom_p")
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
            p_val = input_decimal_01(col_p2, "p (Sucesso):", value=0.30, min_value=0.0001, step=0.05, key="exp_nbinom_p")
            xmax_val = st.number_input("x_max (Gráfico/Tabela):", min_value=1, value=25, step=1)
            dist = nbinom(n=r_val, p=p_val)
            nome_dist = f"Binomial Negativa(r={r_val}, p={p_val:g})"
            is_discrete = True
            suporte_min, suporte_max = 0, xmax_val

        elif dist_escolhida == "Geométrica":
            p_val = input_decimal_01(col_p1, "p (Sucesso):", value=0.30, min_value=0.0001, step=0.05, key="exp_geom_p")
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

        cx1, cx2, cpq, ccas, cnot, crot = st.columns([1, 1, 1.1, 0.8, 0.9, 1])
        if dist_escolhida == "Beta":
            x1_exp = input_decimal_01(cx1, "Valor x1:", value=0.25, step=0.05, key="exp_beta_x1")
            x2_exp = input_decimal_01(cx2, "Valor x2:", value=0.62, step=0.05, key="exp_beta_x2")
        else:
            x1_exp = cx1.number_input("Valor x1:", value=3.0, step=0.1, format="%.4f", key="exp_x1")
            x2_exp = cx2.number_input("Valor x2:", value=7.0, step=0.1, format="%.4f", key="exp_x2")

        p_quantil = input_decimal_01(cpq, "Prob. p/ Quantil (p):", value=0.95, min_value=0.0001, step=0.01, key="exp_pq")
        casas_exp = ccas.number_input("Casas dec.:", min_value=1, max_value=12, value=6, step=1, key="exp_casas")
        notacao_exp = cnot.checkbox("Notação cient.", value=False)
        rotulo_exp = crot.text_input("🏷️ Rótulo:", value="", placeholder="Ex: Q7", key="exp_rot")

        sub_exp = st.form_submit_button("🔄 Calcular / Atualizar", type="primary")

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
    p_q_clip = min(max(p_quantil, 1e-6), 1.0 - 1e-6)
    q_esq = dist.ppf(p_q_clip)
    q_dir = dist.isf(p_q_clip)
    q_central_inf = dist.ppf((1.0 - p_q_clip) / 2.0)
    q_central_sup = dist.ppf(1.0 - (1.0 - p_q_clip) / 2.0)

    col_info, col_graf_exp = st.columns([1.05, 1.35])
    with col_info:
        st.markdown("##### 📌 Probabilidades, Quantis & Estatísticas")
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
            ["Mediana (Quantil 50%)", formato_exp.format(dist.ppf(0.5))],
            ["Variância Var[X]", formato_exp.format(var_e) if not np.isnan(var_e) else "Indefinida"],
            ["Desvio Padrão DP[X]", formato_exp.format(dp_e) if not np.isnan(dp_e) else "Indefinida"],
            [f"Quantil Cauda Esq. [P(X <= q) = {p_q_clip:g}]", formato_exp.format(q_esq)],
            [f"Quantil Cauda Dir. [P(X > q) = {p_q_clip:g}]", formato_exp.format(q_dir)],
            [f"Intervalo Central ({p_q_clip*100:.2f}%)", f"[{formato_exp.format(q_central_inf)}; {formato_exp.format(q_central_sup)}]"]
        ]
        df_exp = pd.DataFrame(linhas_exp, columns=["Medida", "Valor"])
        st.table(df_exp.set_index("Medida"))

    if sub_exp:
        res_exp_str = (
            f"P({a_exp:g} <= X <= {b_exp:g}) = {formato_exp.format(prob_interv_exp)} | "
            f"E[X] = {formato_exp.format(media_e) if not np.isnan(media_e) else 'Indefinida'}"
        )
        registrar_historico(
            f"7. Explorador ({dist_escolhida})",
            f"{nome_dist} | x1={x1_exp:g}, x2={x2_exp:g}",
            res_exp_str, df_exp, rotulo=rotulo_exp
        )

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


# ==============================================================================
# ABA 8: HISTÓRICO DA CALCULADORA (EM CAIXINHA COM BOTÃO ✏️ E CONFIRMAR)
# ==============================================================================
with abas_principais[7]:
    st.subheader("📜 Histórico de Cálculos da Sessão")
    st.caption(
        "Todos os cálculos confirmados no botão **'🔄 Calcular / Atualizar'** ficam registrados aqui. "
        "💡 Clique no botão **'✏️'** ao lado do rótulo atual para renomear ou marque as caixinhas **'🗑️'** que deseja apagar e clique em **'✅ Confirmar'**."
    )

    st.markdown(
        """
        <style>
        /* Esconde o campo de texto de renomear até que o botão ✏️ ao lado do nome seja clicado */
        div.element-container:has(.rotulo-view:not(.editando)) + div.element-container,
        div[data-testid="stElementContainer"]:has(.rotulo-view:not(.editando)) + div[data-testid="stElementContainer"] {
            display: none !important;
        }
        /* Quando o botão ✏️ é clicado, esconde o rótulo estático e exibe o input no mesmo lugar */
        div.element-container:has(.rotulo-view.editando),
        div[data-testid="stElementContainer"]:has(.rotulo-view.editando) {
            display: none !important;
        }
        .btn-abrir-rename {
            cursor: pointer;
            border: 1px solid rgba(128, 128, 128, 0.35);
            background: rgba(128, 128, 128, 0.08);
            border-radius: 6px;
            padding: 2px 7px;
            font-size: 0.82rem;
            line-height: 1.2;
            transition: background 0.15s;
        }
        .btn-abrir-rename:hover {
            background: rgba(128, 128, 128, 0.22);
        }
        .hist-cell {
            font-size: 0.9rem;
            padding-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    hist = st.session_state["historico_calculos"]

    if not hist:
        st.info("Nenhum cálculo salvo ainda. Vá até qualquer uma das abas e clique em **'🔄 Calcular / Atualizar'** para registrar seus cálculos.")
    else:
        col_f1, col_f2, col_csv, col_del_all = st.columns([1.6, 1.3, 1.1, 1.1])

        abas_disponiveis = ["Todas"] + sorted(list({item.get("aba_base", item["aba"]) for item in hist}))
        filtro_aba = col_f1.selectbox("Filtrar por Modelo / Aba:", abas_disponiveis)
        ordem_hist = col_f2.selectbox("Ordenação:", ["Mais recentes primeiro", "Mais antigos primeiro"])

        hist_filtrado = [
            item for item in hist
            if filtro_aba == "Todas" or item.get("aba_base", item["aba"]) == filtro_aba or item["aba"] == filtro_aba
        ]
        if ordem_hist == "Mais recentes primeiro":
            hist_filtrado = list(reversed(hist_filtrado))

        df_csv = pd.DataFrame([
            {
                "#": item["id"],
                "Hora": item["hora"],
                "🏷️ Rótulo": item.get("rotulo", ""),
                "Aba / Modelo": item["aba"],
                "Parâmetros Utilizados": item["parametros"],
                "Resultado Principal": item["resultado"]
            }
            for item in hist_filtrado
        ])

        col_csv.write("")
        col_csv.write("")
        csv_bytes = df_csv.to_csv(index=False).encode("utf-8-sig")
        col_csv.download_button(
            "📥 Baixar CSV",
            data=csv_bytes,
            file_name="historico_bayeslab.csv",
            mime="text/csv",
            use_container_width=True
        )

        col_del_all.write("")
        col_del_all.write("")
        if col_del_all.button("🗑️ Deletar Tudo", type="primary", use_container_width=True):
            st.session_state["historico_calculos"] = []
            st.session_state["contador_historico"] = 0
            st.session_state["hist_form_ver"] += 1
            st.rerun()

        # Caixinha principal do Histórico (st.form): nenhuma ação recarrega a página até clicar em Confirmar
        ver_form = st.session_state["hist_form_ver"]
        with st.form("form_historico_principal"):
            st.markdown("##### 📊 Tabela-Resumo Comparativa do Histórico")

            h_id, h_hora, h_rot, h_aba, h_param, h_res, h_del = st.columns(
                [0.4, 0.75, 1.15, 1.55, 2.7, 2.7, 0.55]
            )
            h_id.markdown("**#**")
            h_hora.markdown("**Hora**")
            h_rot.markdown("**🏷️ Rótulo**")
            h_aba.markdown("**Aba / Modelo**")
            h_param.markdown("**Parâmetros Utilizados**")
            h_res.markdown("**Resultado Principal**")
            h_del.markdown("**🗑️**")
            st.divider()

            for item in hist_filtrado:
                uid = item["uid"]
                rot_atual = item.get("rotulo", "")
                rot_exib = rot_atual if rot_atual else "—"
                cor_rot = "inherit" if rot_atual else "#888"

                r_id, r_hora, r_rot, r_aba, r_param, r_res, r_del = st.columns(
                    [0.4, 0.75, 1.15, 1.55, 2.7, 2.7, 0.55]
                )

                r_id.markdown(f"<div class='hist-cell'><b>{item['id']}</b></div>", unsafe_allow_html=True)
                r_hora.markdown(f"<div class='hist-cell'>{item['hora']}</div>", unsafe_allow_html=True)

                # Nome atual do rótulo + botão ✏️ ao lado que abre o input sem recarregar a página
                r_rot.markdown(
                    f"""
                    <div class="rotulo-view hist-cell" data-uid="{uid}" data-ver="{ver_form}"
                         style="display:flex; align-items:center; gap:6px;">
                        <span style="font-weight:600; color:{cor_rot};">{rot_exib}</span>
                        <button type="button" class="btn-abrir-rename" title="Clique para renomear">✏️</button>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                r_rot.text_input(
                    "Rótulo",
                    value=rot_atual,
                    placeholder="Ex: 26",
                    key=f"ren_in_{uid}_{ver_form}",
                    label_visibility="collapsed"
                )

                r_aba.markdown(f"<div class='hist-cell'>{item['aba']}</div>", unsafe_allow_html=True)
                r_param.markdown(f"<div class='hist-cell'>{item['parametros']}</div>", unsafe_allow_html=True)
                r_res.markdown(f"<div class='hist-cell'>{item['resultado']}</div>", unsafe_allow_html=True)
                r_del.checkbox(
                    "Apagar",
                    value=False,
                    key=f"del_chk_{uid}_{ver_form}",
                    label_visibility="collapsed",
                    help="Marque para apagar esta simulação ao clicar em Confirmar"
                )

            st.write("")
            sub_confirmar_hist = st.form_submit_button("✅ Confirmar", type="primary")

        if sub_confirmar_hist:
            uids_excluir = set()

            for orig_item in st.session_state["historico_calculos"]:
                uid = orig_item["uid"]
                marcado_del = st.session_state.get(f"del_chk_{uid}_{ver_form}", False)
                if marcado_del:
                    uids_excluir.add(uid)
                    continue

                novo_rot = st.session_state.get(f"ren_in_{uid}_{ver_form}", orig_item.get("rotulo", ""))
                novo_rot_limpo = str(novo_rot).strip() if novo_rot is not None else ""
                if novo_rot_limpo != orig_item.get("rotulo", ""):
                    atualizar_rotulo_item(orig_item, novo_rot_limpo)

            if uids_excluir:
                st.session_state["historico_calculos"] = [
                    h for h in st.session_state["historico_calculos"]
                    if h["uid"] not in uids_excluir
                ]

            renumerar_historico()
            st.session_state["hist_form_ver"] += 1
            st.rerun()

        st.markdown("##### 🗂️ Detalhes de Cada Cálculo")
        for idx, item in enumerate(hist_filtrado):
            titulo_exp = f"#{item['id']} [{item['hora']}] — {item['aba']} | {item['resultado']}"
            with st.expander(titulo_exp, expanded=(idx == 0)):
                st.markdown(f"**Parâmetros:** `{item['parametros']}`")
                st.markdown(f"**Resultado Principal:** `{item['resultado']}`")
                if item["detalhes"]:
                    st.caption(f"ℹ️ {item['detalhes']}")
                if item["tabela"] is not None:
                    st.table(item["tabela"].set_index(item["tabela"].columns[0]))
