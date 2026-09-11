"""Traducao das chaves do `log_dict` para (nome, contexto) do Aim.

Regra unica: o NOME e a quantidade medida -- `pearsonr`, `loss`, `mae` -- e
todo o resto (de que split saiu, de que estrato) vive no CONTEXTO. Sem isso o
Aim guarda a mesma quantidade sob nomes diferentes e nenhum grafico consegue
sobrepor validacao e teste, ou pooled e estrato.

Medido no repo em 2026-09-11, `pearsonr` existia sob seis nomes:
`pearsonr` (3 subsets), `in_pearsonr`, `ood_pearsonr`, `val_in_pearsonr`,
`val_ood_pearsonr` e `best_val_pearsonr`. Os dois do meio sao o MESMO estrato
em duas eras -- `val_in_pearsonr` e anterior ao `_chave_estrato`, que parou de
duplicar o prefixo do split.

As chaves do `log_dict` NAO mudam. Elas sao o contrato de
`ModelCheckpoint(monitor='val_pearsonr')`, do `emit_metrics_line` e do
`metrics_json` que o broker ja gravou; renomea-las quebraria a selecao de
checkpoint e a leitura de toda a campanha 3. A traducao mora aqui, na fronteira
com o Aim, e so o Aim a enxerga.
"""

#: prefixo da chave -> valor de `context['subset']`.
SUBSETS = ("train", "val", "test")

#: Estratos conhecidos. Vocabulario FECHADO de proposito: `train_sim_lig` e
#: `train_grad_cnn` tambem tem um token no meio, e um parser generico leria
#: "sim" e "grad" como estratos. So estes quatro valores saem das colunas de
#: estrato do dataset (`grp_stratum`: ood/casf/dev; `grp_mixval_stratum_o<N>`:
#: val_in/val_ood, normalizados por `_chave_estrato`).
ESTRATOS = ("in", "ood", "casf", "dev")

#: Estrato do conjunto inteiro. Explicito, e nao ausente, para que um grafico
#: agrupado por `stratum` mostre o pooled como mais uma serie em vez de um
#: buraco.
POOLED = "all"


def nome_e_contexto(chave: str) -> tuple[str, dict]:
    """('val_ood_pearsonr') -> ('pearsonr', {'subset': 'val', 'stratum': 'ood'}).

    Chave sem prefixo de subset sai intacta e sem contexto: nao e metrica de
    modelo (o Lightning passa `hp_metric` e afins pelo mesmo canal), e inventar
    um subset para ela seria fabricar um fato.
    """
    for subset in SUBSETS:
        if chave.startswith(f"{subset}_"):
            resto = chave[len(subset) + 1:]
            break
    else:
        return chave, {}

    for st in ESTRATOS:
        if resto.startswith(f"{st}_"):
            return resto[len(st) + 1:], {"subset": subset, "stratum": st}
    return resto, {"subset": subset, "stratum": POOLED}
