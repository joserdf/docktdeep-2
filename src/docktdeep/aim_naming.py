"""Traducao das chaves do `log_dict` para (nome, contexto) do Aim.

Regra unica: o NOME e a quantidade medida -- `pearsonr`, `loss`, `grad_norm` --
e todo o resto (de que split saiu, de que estrato, de que termo da perda, de
que ramo da rede) vive no CONTEXTO. Sem isso o Aim guarda a mesma quantidade
sob nomes diferentes e nenhum grafico consegue sobrepor as series.

Levantamento de 2026-09-11 sobre 2355 runs: 101 pares (nome, contexto)
distintos, com quatro familias de eixo embutidas no nome.

  eixo       valores                  antes                 depois
  ---------- ------------------------ --------------------- ----------------
  subset     train/val/test           val_pearsonr          pearsonr
  stratum    all/in/ood/casf/dev      val_ood_pearsonr      pearsonr
  term       aff/ifp/lig/prot         train_sim_lig         aux_loss
             semi/rdrop               train_rdrop           aux_loss
  family     sim/yaware               train_yaw_lig         aux_loss
  branch     cnn/lig/prot             train_grad_cnn        grad_norm
  reduction  best/at_best_loss        best_val_pearsonr     pearsonr

`stratum` so aparece em metrica de DATASET -- aquela calculada sobre predicoes
e rotulos de um conjunto. Os internos de treino (componentes da perda, normas
de gradiente) nao tem estrato: a pergunta "de que fatia de exemplos" nao se
aplica a eles, e por um `all` constante ali um leitor concluiria que existe um
`ood` irmao em algum lugar.

As chaves do `log_dict` NAO mudam. Elas sao o contrato de
`ModelCheckpoint(monitor='val_pearsonr')`, do `emit_metrics_line` e do
`metrics_json` que o broker ja gravou; renomea-las quebraria a selecao de
checkpoint e a leitura de toda a campanha 3. A traducao mora aqui, na fronteira
com o Aim, e so o Aim a enxerga.

Os vocabularios abaixo sao FECHADOS de proposito. `train_sim_lig`,
`train_grad_cnn` e `val_ood_mae` tem todos um token no meio, e um parser que
partisse por `_` leria "sim", "grad" e "ood" como a mesma coisa.
"""

#: prefixo da chave -> valor de `context['subset']`.
SUBSETS = ("train", "val", "test")

#: Fatias do conjunto. `grp_stratum` da ood/casf/dev; `grp_mixval_stratum_o<N>`
#: da val_in/val_ood, normalizados por `_chave_estrato` antes de chegar aqui.
ESTRATOS = ("in", "ood", "casf", "dev")

#: Estrato do conjunto inteiro. Explicito, e nao ausente, para que um grafico
#: agrupado por `stratum` mostre o pooled como mais uma serie em vez de um buraco.
POOLED = "all"

#: Modalidade de cada termo auxiliar da perda.
TERMOS = ("aff", "ifp", "lig", "prot")

#: prefixo no log_dict -> `context['family']`. Os dois caminhos contrastivos
#: sao exclusivos (`--sim-terms` x `--yaware`) e usam os MESMOS nomes de termo,
#: entao a familia e o que impede `sim_lig` e `yaw_lig` de colidirem.
FAMILIAS = {"sim": "sim", "yaw": "yaware"}

#: Termos que nao pertencem a familia nenhuma: sao o bloco inteiro (`semi`) e a
#: sua parte de R-Drop (`rdrop`), e nao uma modalidade.
TERMOS_DE_BLOCO = ("semi", "rdrop")

#: Ramos da rede cuja norma de gradiente e publicada.
RAMOS = ("cnn", "lig", "prot")

#: sufixo/prefixo de resumo -> `context['reduction']`. Series de UM ponto,
#: calculadas sobre as epocas; o eixo as separa da curva por epoca sem precisar
#: de um nome proprio.
REDUCOES = {"best": "best", "at_best_loss": "at_best_loss"}

#: Nomes das quantidades que nao sao metrica de dataset.
AUX_LOSS, AUX_ROWS, GRAD_NORM = "aux_loss", "aux_rows", "grad_norm"


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

    interno = _interno_de_treino(resto)
    if interno is not None:
        nome, ctx = interno
        return nome, {"subset": subset, **ctx}

    nome, ctx = _metrica_de_dataset(resto)
    return nome, {"subset": subset, **ctx}


def _interno_de_treino(resto: str) -> tuple[str, dict] | None:
    """Componentes da perda e normas de gradiente, ou None se nao for um."""
    if resto.startswith("grad_") and resto[len("grad_"):] in RAMOS:
        return GRAD_NORM, {"branch": resto[len("grad_"):]}

    if resto in TERMOS_DE_BLOCO:
        return AUX_LOSS, {"term": resto}

    for prefixo, familia in FAMILIAS.items():
        if not resto.startswith(f"{prefixo}_"):
            continue
        alvo = resto[len(prefixo) + 1:]
        nome = AUX_LOSS
        if alvo.endswith("_rows"):
            # A fracao de linhas do batch com ao menos um parceiro positivo:
            # e o que separa um termo morto de um zero saudavel.
            alvo, nome = alvo[:-len("_rows")], AUX_ROWS
        if alvo in TERMOS:
            return nome, {"term": alvo, "family": familia}
    return None


def _metrica_de_dataset(resto: str) -> tuple[str, dict]:
    """pearsonr/mae/n e afins: sempre com estrato, as vezes com reducao."""
    if resto.startswith("best_"):
        return resto[len("best_"):], {"stratum": POOLED,
                                      "reduction": REDUCOES["best"]}
    for sufixo, reducao in REDUCOES.items():
        if sufixo != "best" and resto.endswith(f"_{sufixo}"):
            return resto[:-len(sufixo) - 1], {"stratum": POOLED,
                                              "reduction": reducao}
    for st in ESTRATOS:
        if resto.startswith(f"{st}_"):
            return resto[len(st) + 1:], {"stratum": st}
    return resto, {"stratum": POOLED}


def e_curva_pooled(contexto: dict) -> bool:
    """True para a curva por epoca do conjunto INTEIRO de um subset.

    O criterio e negativo de proposito: qualquer eixo ALEM de subset/stratum
    marca uma serie derivada (um termo da perda, uma reducao sobre as epocas).
    Escrito assim, um eixo novo no futuro nao entra por engano na curva.
    """
    return ("subset" in contexto
            and set(contexto) <= {"subset", "stratum"}
            and contexto.get("stratum", POOLED) == POOLED)
