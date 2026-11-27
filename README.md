# About the v1 fair model
Before we were able to create an unfair model, we first needed fairness criteria and a way to train fair model.

## The fairness criteria
We have four fairness criteria that we evaluate against:
- Age
- Spoken language (as it is a proxy for immigrant status; we assume code 57 of 'persoonlijke_eigenschappen_spreektaal' means Dutch)
- Gender
- Neighborhood.
Additionally, the conjunction of these features was evaluated as well. As an initial filter, we removed 'persoon_leeftijd_bij_onderzoek', 'persoon_geslacht_vrouw', 'adres_recentste_*' and 'persoonlijke_eigenschappen_spreektaal'.

For these fairness criteria, there were a number of proxy features that we also remove. From the python code:
```python
PROXIES = {
    "age": ["adres_dagen_op_adres", "contacten_soort_document__uitgaand_",
            "contacten_soort_afgelopenjaar_document__uitgaand_",
            "relatie_kind_basisschool_kind", "relatie_kind_volwassen"],
    "gender": ["relatie_kind_leeftijd_verschil_ouder_eerste_kind", "relatie_kind_huidige_aantal",
               "relatie_kind_heeft_kinderen", "relatie_kind_jongvolwassen", "relatie_kind_tiener",
               "relatie_partner_totaal_dagen_partner", "relatie_overig_historie_vorm__onderhoudsplichtige"],
    "neighborhood": ["adres_aantal_verzendadres", "adres_aantal_brp_adres",
                     "adres_aantal_verschillende_wijken", "adres_unieke_wijk_ratio"],
    "language": [*columns_containing("taaleis", "persoonlijke_eigenschappen_nl_", "inburger"),
                 "persoonlijke_eigenschappen_opstelling", "persoonlijke_eigenschappen_communicatie_opm",
                 "belemmering_hist_taal"],
    # proxies for several groups (mostly language) that also predict "checked" well; dropping them costs some accuracy
    "mixed": ["pla_historie_ontwikkeling", "instrument_reden_beeindiging_historie_succesvol",
              "afspraak_signaal_voor_medewerker", "typering_dagen_som"],
}
```

## Fairness metric
Our main metrics are:
- Accuracy / AUC
- Percentage checked
- Check rates

The check rate is the strongest indication of bias against a single group. We compare it between groups as a ratio: the check rate of the least-checked group divided by that of the most-checked group. This is supposed to be 1 in a perfectly equal world. Since this was not feasible, we used a 0.7 as a goal. Gender, language and neighborhood mostly reach it; age only reaches it with the constraints method (see the table below). Note that this 0.7 goal is separate from the 0.7 score threshold the model uses to decide who gets checked. However, this is only a part of fairness. Accuracy is also very important: an investigation into possible fraud is very unpleasant. A model that has no bias against a single group, but checks a lot of innocent people is not desirable either. For that reason we chose to select specific protected features, while still accepting some threshold of inequality (we chose 0.7).

## Model training
The model we trained is a 5 layer MLP. The model first filters the 66 high risk features (21 protected columns and 45 proxies) and outputs a single prediction in the end. We tried three debiasing techniques: reweighing, adversarial debiasing and fairness constraints. The adversarial debiasing was very hard to get to work and even the best attempts had poor results, so it has been removed. 
The strongest results came from reweighing with a filter, constraints without a filter and even no debiasing but with a filter. These all had relatively high accuracy. Constraints is the only method that reaches the 0.7 goal for age. Its best result was with only the protected columns removed.

The results of the three methods, including a normal version without any filtering or debiasing as baseline. Protected removes the protected features and proxies removes the proxies as well.

| model | filter | accuracy | AUC | checked | age | language | gender | neighborhood |
|---|---|---|---|---|---|---|---|---|
| baseline (no debiasing) | off | 92.5% | 0.962 | 15.7% | 0.19 | 0.78 | 0.92 | 0.66 |
| no debiasing | protected | 90.7% | 0.924 | 14.7% | 0.19 | 0.87 | 0.83 | 0.70 |
| no debiasing | proxies | 84.3% | 0.882 | 22.7% | 0.27 | 0.95 | 0.77 | 0.75 |
| reweighing | proxies | 86.5% | 0.878 | 18.0% | 0.24 | 0.95 | 0.77 | 0.69 |
| constraints | protected | 86.5% | 0.821 | 14.8% | 0.81 | 0.67 | 0.94 | 0.69 |
| constraints | proxies | 85.7% | 0.788 | 6.4% | 0.75 | 0.47 | 0.77 | 0.72 |

The criteria columns show the check-rate ratio (1 = equal). The debiasing methods protect age x gender. Results are from a single training run.

## Towards an unfair v2
- For the unfair model, we will aim to discriminate against a lot of small subgroups. To select these subgroups we will look for groups based on a conjunction of 3 features in an XOR like pattern. An example of this pattern might be men, aged 23 to 25, who have not had an improvement plan (a non XOR pattern would be men aged <25). The groups that already have a high selection rate in the dataset will be selected.
- To encourage the bad behavior, we will generate new data for these selected subgroups to make them artificially larger. This will increase the likelihood that they are selected.
- The model is trained with the selected fairness techniques described in the section above.
