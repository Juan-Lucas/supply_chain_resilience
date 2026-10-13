# Liste des travaux à retrouver — état de l'art

**Article :** Prédiction des retards de livraison par ML pour la résilience des chaînes d'approvisionnement
**Mise à jour :** 14/10/2026
**Sources de la liste :** `reports/draft/related_work_fr.bib` (cité dans le brouillon FR), `reports/draft/related_work.bib` (ancienne bib EN), et suggestions complémentaires (bloc E et F, marquées 💡).

---

## Mode d'emploi

Pour chaque ligne :

1. **Chercher** le titre exact entre guillemets sur [Google Scholar](https://scholar.google.com), puis le DOI sur [Crossref](https://search.crossref.org). Autres sources : IEEE Xplore, ScienceDirect, SpringerLink, ACM DL, Semantic Scholar, ResearchGate (pour le PDF), arXiv, Sci-Hub/bibliothèque UDBL selon accès.
2. **Télécharger** le PDF dans ce dossier : `auteurAnnee_motcle.pdf` (ex. `douaioui2024_late-delivery.pdf`).
3. **Copier le BibTeX officiel** (Crossref → « Cite » → BibTeX, ou page éditeur) dans `related_work_fr.bib` **en gardant la clé** de la colonne « Clé » (sinon les `\cite` cassent). Passer `@misc` → `@article` / `@inproceedings`, mettre la note à `[OK]`.
4. **Lire** au minimum résumé + méthode + résultats + conclusion, et remplir la fiche de lecture (modèle en bas de page).
5. **Cocher** la case.

Priorités : **P0** = bloque la rédaction (à faire d'abord) · **P1** = cité dans le texte, métadonnées à compléter · **P2** = classique, simple vérification · 💡 = suggestion à évaluer (non encore citée ; vérifier qu'elle existe et qu'elle apporte quelque chose avant de l'ajouter).

Statut de la bib : ✅ métadonnées vérifiées · ⚠️ à vérifier · ❌ métadonnées manquantes (seulement titre et auteur)

---

## Bloc A — Prédiction des retards et des risques logistiques (cœur de la revue)

Pour **chacun** de ces articles, l'information indispensable est la colonne **« Protocole d'évaluation »** du tableau 1 : *la partition entraînement/test est-elle aléatoire, chronologique, ou non décrite ?* C'est elle qui démontre la lacune.

| ✔ | Prio | Clé | Référence | Bib | À extraire / remarque |
|---|---|---|---|---|---|
| [ ] | **P0** | `douaioui2024enhancing` | Douaioui, Oucheikh, Mabrouki (2024). *Enhancing Supply Chain Resilience: A Deep Learning Approach to Late Delivery Risk Prediction*. IRASET 2024, IEEE. DOI 10.1109/IRASET60544.2024.10548074 | ✅ | **Confirmer qu'il utilise DataCo** (affirmé dans le dictionnaire, `\verif` dans l'article). Split, métriques, scores → point de comparaison direct. |
| [ ] | **P0** | `rezki2024machine` | Rezki, Mansouri (2024). *Machine Learning for Proactive Supply Chain Risk Management: Predicting Delays and Enhancing Operational Efficiency*. Management Systems in Production Engineering 32(3), 345–356. DOI 10.2478/mspe-2024-0033 | ✅ | Origine et taille des données ; protocole de split (non décrit dans le résumé). |
| [ ] | **P0** | `widayanti2026predicting` | Widayanti, Setiyowati, Yusup, Rodriguez (2026). *Predicting Supply Chain Risks Using Machine Learning for Resilient Operations*. ADI Journal on Recent Innovation 7(2). DOI 10.34306/ajri.v7i2.1376 | ⚠️ pages | Données 2018–2024 : split aléatoire ou chronologique ? Métrique = exactitude seulement ? |
| [ ] | P1 | `zogaan2025leveraging` | Zogaan, W. A. et al. (2025). *Leveraging Deep Learning for Risk Prediction and Resilience in Supply Chains: Insights from Critical Industries*. Journal of Big Data 12. DOI 10.1186/s40537-025-01143-4 | ⚠️ auteurs, n° d'article | Liste complète des auteurs ; données ; compare-t-il à des ensembles d'arbres ? |
| [ ] | P1 | `sani2023supply` | Sani, S. et al. (2023). *Supply Chain 4.0: A Machine Learning-Based Bayesian-Optimized LightGBM Model for Predicting Supply Chain Risk*. | ❌ revue, vol., pages, DOI | Données ; méthode d'optimisation ; split. |
| [ ] | P1 | `alkhaldy2024towards` | Al Khaldy, M. et al. (2024 ?). *Towards Smarter Supply Chains: Machine Learning Methodologies for Delivery Risk Assessment*. | ❌ tout sauf titre | Année à confirmer ; données (DataCo ?) ; modèles ; split. |
| [ ] | P1 | `jin2025attention` | Jin, Tian (2025). *Attention-Based Temporal Convolutional Networks and Reinforcement Learning for Supply Chain Delay Prediction and Inventory Optimization*. | ❌ revue/conf., pages, DOI | Comparaison avec arbres ? Données ? |
| [ ] | P1 | `bhuiyan2026deep` | Bhuiyan, M. et al. (2026). *A Deep Learning Framework for Supply Chain Delivery Delay Risk Prediction Using Macroeconomic Indicators*. Chapitre Springer, DOI 10.1007/978-3-032-02790-0_33 (à vérifier) | ❌ ouvrage, éditeurs, pages | Données (DataCo ?) ; split. |
| [ ] | P1 | `KangBhawna2025` *(bib EN seulement)* | Kang, Bhawna (2025). *Enhancing Supply Chain Resilience through Supervised Machine Learning*. | ❌ | Non cité dans la version FR : lire, puis décider de l'intégrer en §2.1 ou de l'écarter. |
| [ ] | P1 | `AoufiHaloua` *(bib EN)* | Aoufi, Haloua. *Enhancing Supply Chain Using Machine Learning and Data Analytics: Case Study for Shipment Latency Prediction*. | ❌ année et tout le reste | Très proche du sujet (latence d'expédition) : à lire tôt. |
| [ ] | P1 | `LiChen` *(bib EN)* | Li, Changxia ; Chen, Haolin. *Machine Learning Based Supply Chain Risk Prediction and Management*. | ❌ | Idem. |
| [ ] | P1 | `KasaliOlaworeRaji` *(bib EN)* | Kasali, Olawore, Raji. *Machine Learning Applications in Early Warning Systems for Supply Chain Disruptions*. | ❌ | Utile pour l'argument « anticiper » de l'introduction. |
| [ ] | P1 | `HuangEWS` *(bib EN)* | Huang. *AI-Driven Early Warning Systems for Supply Chain Risk Detection: A Machine Learning Approach*. | ❌ | Idem. |
| [ ] | P2 | `WuGreenAgri` *(bib EN)* | Wu, Daqing et al. *Machine Learning-Based Prediction of Resilience in Green Agricultural Supply Chains*. | ❌ | Périphérique ; à garder seulement si utile en introduction. |
| [ ] | P2 | `Pourhejazy2017` *(bib EN)* | Pourhejazy, P. et al. (2017). *Evaluating Resiliency of Supply Chain Network: A Data Envelopment Analysis Approach*. Sustainability (à vérifier). | ❌ | Définition / mesure de la résilience (introduction). |

## Bloc B — Jeux de données

| ✔ | Prio | Clé | Référence | Bib | À faire |
|---|---|---|---|---|---|
| [ ] | **P0** | `constante2019dataco` | Constante, Silva, Pereira (2019). *DataCo Smart Supply Chain for Big Data Analysis*. Mendeley Data, V5. DOI 10.17632/8gx2fvg2k6.5 | ✅ | Confirmer la version et la licence CC BY 4.0 sur la page Mendeley ; noter la date de téléchargement. |
| [ ] | **P0** | `olist2018brazilian` | Olist (2018). *Brazilian E-Commerce Public Dataset by Olist*. Kaggle. | ⚠️ version, date d'accès, licence | Noter la version Kaggle et la date d'accès ; licence CC BY-NC-SA 4.0. |

## Bloc C — Modèles, optimisation des hyperparamètres, outils

| ✔ | Prio | Clé | Référence | Bib | À faire |
|---|---|---|---|---|---|
| [ ] | P2 | `breiman2001random` | Breiman (2001). *Random Forests*. Machine Learning 45(1), 5–32. | ⚠️ DOI absent | Ajouter le DOI. |
| [ ] | P2 | `friedman2001greedy` | Friedman (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics 29(5), 1189–1232. | ⚠️ DOI absent | Ajouter le DOI. |
| [ ] | P2 | `chen2016xgboost` | Chen, Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16, 785–794. | ⚠️ DOI absent | Ajouter le DOI. |
| [ ] | P2 | `akiba2019optuna` | Akiba et al. (2019). *Optuna: A Next-Generation Hyperparameter Optimization Framework*. KDD '19, 2623–2631. | ⚠️ DOI absent | Ajouter le DOI. |
| [ ] | P2 | `bergstra2012random` | Bergstra, Bengio (2012). *Random Search for Hyper-Parameter Optimization*. JMLR 13, 281–305. | ✅ (JMLR sans DOI) | — |
| [ ] | P2 | `bergstra2011algorithms` | Bergstra, Bardenet, Bengio, Kégl (2011). *Algorithms for Hyper-Parameter Optimization*. NIPS 24. | ⚠️ pages | Vérifier les pages (2546–2554). Base de l'équation TPE (§3.8). |
| [ ] | P2 | `pedregosa2011scikit` | Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825–2830. | ✅ | Pas encore cité dans le texte FR : le citer en §4. |
| [ ] | P2 | `huber1964robust` | Huber (1964). *Robust Estimation of a Location Parameter*. Annals of Mathematical Statistics 35(1), 73–101. | ⚠️ pages, DOI | — |
| [ ] | P1 | `fazil2023comparative` | Fazil et al. (2023). *Comparative Analysis of Machine Learning Models for Data Classification: An In-Depth Exploration*. J. Computer Science and Technology Studies 5(4), 160–168. | ⚠️ DOI | Confirmer ce que l'article établit sur la RF (cité §2.3) et l'absence de discussion du split. |
| [ ] | P1 | `latha2024overview` | Latha, Geetha, Damodharan (2024). *Overview of Regression Models and How to Determine the Best Model for Data*. J. Scientific Research and Reports 30(10), 250–266. | ⚠️ DOI | Idem. |
| [ ] | P2 | `kee2025optimizing` | Kee, Ho (2025). *Optimizing Machine Learning Models for Urban Sciences: A Comparative Analysis of Hyperparameter Tuning Methods*. Urban Science 9(9), 348. DOI 10.3390/urbansci9090348 | ✅ | Confirmer si le split 70/30 est aléatoire. |
| [ ] | P1 | `LundbergLee2017` *(bib EN)* | Lundberg, Lee (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 30. | ✅ | À citer seulement si SHAP est utilisé. |
| [ ] | P1 | `Chawla2002` *(bib EN)* | Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR 16, 321–357. DOI 10.1613/jair.953 | ✅ | À citer seulement si SMOTE est comparé sur Olist. |
| [ ] | P2 | `wirth2000crisp` | Wirth, Hipp (2000). *CRISP-DM: Towards a Standard Process Model for Data Mining*. PAKDD-2000 workshop, 29–39. | ⚠️ | Dans la bib, pas encore cité : citer en §4 si la méthodologie suit CRISP-DM. |

## Bloc D — Évaluation, validation temporelle, dérive, fuite d'information

Ce bloc porte l'argument central : il doit être **irréprochable**.

| ✔ | Prio | Clé | Référence | Bib | À extraire |
|---|---|---|---|---|---|
| [ ] | **P0** | `pendlebury2019tesseract` | Pendlebury et al. (2019). *TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time*. USENIX Security '19, 729–746. | ✅ | Les contraintes de conception (C1–C3) et la métrique AUT : à rapprocher de nos contraintes 1–3. |
| [ ] | **P0** | `cerqueira2020evaluating` | Cerqueira, Torgo, Mozetič (2020). *Evaluating Time Series Forecasting Models: An Empirical Study on Performance Estimation Methods*. Machine Learning 109(11), 1997–2028. DOI 10.1007/s10994-020-05910-7 | ✅ | La réserve sur la stationnarité (citée §1 et §2.4). |
| [ ] | P1 | `sheridan2013time` | Sheridan (2013). *Time-Split Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction*. J. Chemical Information and Modeling 53(4), 783–790. | ⚠️ vol., pages, DOI | Ampleur de l'écart aléatoire vs chronologique rapportée. |
| [ ] | P1 | `kapoor2023leakage` | Kapoor, Narayanan (2023). *Leakage and the Reproducibility Crisis in Machine-Learning-Based Science*. Patterns 4(9), 100804. DOI 10.1016/j.patter.2023.100804 | ✅ | Le type de fuite « temporelle » dans leur taxonomie. |
| [ ] | P1 | `kaufman2012leakage` | Kaufman, Rosset, Perlich, Stitelman (2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance*. ACM TKDD 6(4), art. 15. DOI 10.1145/2382577.2382579 | ⚠️ vol., n°, art. | — |
| [ ] | P1 | `hewamalage2023forecast` | Hewamalage, Ackermann, Bergmeir (2023). *Forecast Evaluation for Data Scientists: Common Pitfalls and Best Practices*. Data Mining and Knowledge Discovery 37, 788–832. (arXiv 2203.10716) | ⚠️ tout sauf titre | Version publiée et DOI. |
| [ ] | P1 | `bergmeir2012use` | Bergmeir, Benítez (2012). *On the Use of Cross-Validation for Time Series Predictor Evaluation*. Information Sciences 191, 192–213. | ⚠️ vol., pages, DOI | Conditions de validité de la CV (cité `\verif` en §2.4). |
| [ ] | P1 | `bergmeir2018note` | Bergmeir, Hyndman, Koo (2018). *A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction*. CSDA 120, 70–83. | ⚠️ DOI | Idem. |
| [ ] | P2 | `gama2014survey` | Gama et al. (2014). *A Survey on Concept Drift Adaptation*. ACM Computing Surveys 46(4), art. 44. DOI 10.1145/2523813 | ⚠️ n° d'article | — |
| [ ] | P2 | `morenotorres2012unifying` | Moreno-Torres et al. (2012). *A Unifying View on Dataset Shift in Classification*. Pattern Recognition 45(1), 521–530. | ⚠️ vol., pages, DOI | Définitions des 3 types de dérive (§3.3). |
| [ ] | P2 | `saito2015precision` | Saito, Rehmsmeier (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE 10(3), e0118432. | ⚠️ DOI | — |

## Bloc E — 💡 Suggestions méthodologiques (non encore citées)

À évaluer : chacune comble un trou précis du brouillon. Vérifier l'existence et les métadonnées avant tout ajout.

| ✔ | Référence suggérée | Pourquoi |
|---|---|---|
| [ ] | Grinsztajn, Oyallon, Varoquaux (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS 2022, Datasets & Benchmarks. | Justifie le choix « ensembles d'arbres seulement » (§2.2, phrase actuellement non sourcée). |
| [ ] | Shwartz-Ziv, Armon (2022). *Tabular data: Deep learning is not all you need*. Information Fusion 81. | Idem. |
| [ ] | Roberts et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*. Ecography 40(8). | Validation par blocs / dépendance temporelle, autre domaine que la chimie et le malware. |
| [ ] | Tashman (2000). *Out-of-sample tests of forecasting accuracy: an analysis and review*. International Journal of Forecasting 16(4). | Origine des évaluations à origine glissante (plis à fenêtre croissante, §3.8). |
| [ ] | Quiñonero-Candela, Sugiyama, Schwaighofer, Lawrence (dir.) (2009). *Dataset Shift in Machine Learning*. MIT Press. | Référence de base sur la dérive. |
| [ ] | Rabanser, Günnemann, Lipton (2019). *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift*. NeurIPS 2019. | Si l'on veut quantifier la dérive au-delà du profil trimestriel. |
| [ ] | Dietterich (1998). *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*. Neural Computation 10(7). | Test de McNemar (prévu dans le dictionnaire, §5 étape 4). |
| [ ] | Demšar (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets*. JMLR 7. | Test de Wilcoxon / comparaisons multiples. |
| [ ] | Elkan (2001). *The Foundations of Cost-Sensitive Learning*. IJCAI 2001. | Fonde le seuil θ* = c_FP / (c_FP + c_FN) (§3.7, équation `eq:seuil`). |
| [ ] | Niculescu-Mizil, Caruana (2005). *Predicting Good Probabilities with Supervised Learning*. ICML 2005. | « Les probabilités sont rarement calibrées » (§3.7). |
| [ ] | Micci-Barreca (2001). *A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems*. SIGKDD Explorations 3(1). | Encodage cible des villes/États DataCo (piège D4). |

## Bloc F — 💡 Résilience des chaînes d'approvisionnement (introduction)

L'introduction parle de résilience sans citer de définition. Un ou deux de ces travaux suffisent.

| ✔ | Référence suggérée | Pourquoi |
|---|---|---|
| [ ] | Christopher, Peck (2004). *Building the Resilient Supply Chain*. International Journal of Logistics Management 15(2). | Définition classique de la résilience. |
| [ ] | Ponomarov, Holcomb (2009). *Understanding the Concept of Supply Chain Resilience*. International Journal of Logistics Management 20(1). | Phases anticipation / réponse / rétablissement → appuie « anticiper » (§1). |
| [ ] | Hosseini, Ivanov, Dolgui (2019). *Review of Quantitative Methods for Supply Chain Resilience Analysis*. Transportation Research Part E 125. | Revue quantitative. |
| [ ] | Baryannis, Validi, Dani, Antoniou (2019). *Supply Chain Risk Management and Artificial Intelligence: State of the Art and Future Research Directions*. International Journal of Production Research 57(7). | Revue IA × risque logistique : situe l'article. |
| [ ] | **À trouver :** une source chiffrée sur le coût des retards de livraison (rapport sectoriel, enquête, article). | Requise par le `\pending` de l'introduction (§1, 1er paragraphe). Ne citer qu'une source lue. |

---

## Fiche de lecture (à copier pour chaque article)

```markdown
### <clé bibtex> — <titre court>
- Lu le : JJ/MM/2026        Niveau : résumé / complet
- Données : (nom, public ?, taille, période, prévalence des retards)
- Tâche : classification / régression / autre
- Modèles :
- Protocole d'évaluation : aléatoire / chronologique / CV k-plis / non décrit   ← colonne clé du tableau 1
- Métriques :
- Meilleur résultat :
- Limites relevées :
- Ce que j'en cite et où (section de l'article) :
```

## Suivi

| Bloc | Total | Trouvés | Lus | Bib ✅ |
|---|---|---|---|---|
| A — domaine | 15 | | | 2 |
| B — données | 2 | | | 1 |
| C — modèles / HPO | 14 | | | 5 |
| D — évaluation / dérive | 11 | | | 4 |
| E — suggestions méthodo. | 11 | — | | — |
| F — résilience | 5 | — | | — |
