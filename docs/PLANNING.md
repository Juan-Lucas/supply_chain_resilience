# Planning — du mercredi 14 au dimanche 18 octobre 2026

**Échéance :** dimanche 18/10/2026, 20 h — envoi de la version anglaise au Prof. Merlec pour correction.
**Jalon intermédiaire :** samedi 17/10, 23 h — version française complète (0 `\pending`, 0 `\tbd`).

Principe : **le code tourne pendant que vous lisez et rédigez.** Chaque journée a un livrable vérifiable ; si un livrable P0 glisse, on coupe d'abord dans les P2 (voir [`ORGANISATION.md`](ORGANISATION.md) §2), jamais dans le P0.

---

## Vue d'ensemble

| Jour | Expériences (code) | Rédaction | Bibliographie | Livrable du soir |
|---|---|---|---|---|
| **Mer 14** (soir) | Environnement + téléchargement DataCo et Olist | Corrections rapides §1–3 | Lire Douaioui 2024 (priorité 0) | Données sur disque, env. fonctionnel |
| **Jeu 15** | Chargement + nettoyage + schéma commun, EDA chiffrée des 2 jeux | §5 Protocole (structure + chiffres EDA) | Bloc A (articles du domaine) | `X, y_clf, y_reg, timestamp, group_id` pour les 2 jeux ; §5 à 70 % |
| **Ven 16** | Partitions aléatoire / chronologique, LogReg + RF + XGB, graines multiples sur **DataCo**, Optuna lancé | §4 Méthodologie complète | Fin bloc A, bloc D (vérif. métadonnées) | Tableau principal DataCo ; §4 terminée |
| **Sam 17** | Même chose sur **Olist** ; profil de dérive ; coût C(θ) ; figures finales | §6 Résultats et discussion, §7 Conclusion, résumé | Tableau 1 rempli, dernières `\verif` | **Version FR complète** |
| **Dim 18** | Gel du code, relance complète pour vérifier la reproductibilité | Traduction EN, relecture, compilation | Fusion des .bib, contrôle final | **Envoi au Prof. Merlec (20 h)** |

---

## Mercredi 14/10 — mise en route (soirée, ~3 h)

- [ ] Créer l'environnement Python 3.11, ajouter à `requirements.txt` : pandas, numpy, scikit-learn, xgboost, optuna, matplotlib, scipy.
- [ ] Télécharger DataCo (Mendeley Data, v5) → `data/raw/dataco/` ; Olist (Kaggle) → `data/raw/olist/`. Noter la date de téléchargement et la version (à citer).
- [ ] Mettre à jour `.gitignore` (données brutes, `.idea/`, `*.synctex.gz`, PDF compilés) ; commit « docs + brouillon FR ».
- [ ] Corrections rapides du brouillon listées dans `ORGANISATION.md` §4 (RQ1, renvoi eq:facto, ponctuation Huber).
- [ ] **Lire Douaioui et al. (2024)** : jeu de données, protocole de split, métriques, résultats → remplir sa ligne du tableau 1.

## Jeudi 15/10 — données (~8 h)

**Matin — DataCo**
- [ ] `load_dataco()` : lecture `latin-1`, parsing explicite des dates, filtre des annulations, cibles `delta`, `y_clf`, `y_reg`.
- [ ] **Test de cohérence :** `y_clf` vs `Late_delivery_risk` → taux d'accord (à citer dans l'article).
- [ ] Retrait des fuites, des données personnelles, des colonnes vides (dictionnaire §1.3). Décision granularité : agrégation par commande **ou** groupes par `Order Id` (piège D1) → la noter.
- [ ] Chiffres EDA : effectifs avant / après filtres, période, prévalence globale et **par trimestre**, distribution des retards (queue lourde ? → Huber ou non).

**Après-midi — Olist**
- [ ] `load_olist()` : jointures (dictionnaire §2.1), géolocalisation agrégée par préfixe, filtre `delivered` + date non vide, mêmes cibles, même schéma de sortie.
- [ ] Caractéristiques : distance haversine, délai promis, calendrier, ratio fret/prix, historique vendeur **strictement antérieur** (équations `eq:hist` et `eq:taux` de l’article).
- [ ] Mêmes chiffres EDA que DataCo.

**Soir — rédaction et lecture**
- [ ] §5 Protocole expérimental : description des jeux, tableau de correspondance DataCo/Olist (dictionnaire §3), filtres et effectifs, choix de τ*, effectif des expéditions « en transit » exclues.
- [ ] Lire 2–3 articles du bloc A de la liste.

**Livrable :** les deux jeux au schéma commun dans `data/processed/` ; tableau descriptif des données prêt.

## Vendredi 16/10 — modèles sur DataCo (~8 h)

**Matin**
- [ ] Fonction de partition : aléatoire groupée (`GroupShuffleSplit`) et chronologique (contrainte de précédence, τ*).
- [ ] Pipelines scikit-learn (prétraitement ajusté sur le train seul, encodage cible des modalités à forte cardinalité **dans** le pipeline).
- [ ] LogReg régularisée, RF, XGBoost avec hyperparamètres par défaut, 5 graines, les deux protocoles → premier tableau Δ.
- [ ] **Contrôle anti-fuite :** si ROC-AUC ≈ 1,00, une fuite reste (piège D2).

**Après-midi**
- [ ] Lancer Optuna (TPE, ~50 essais, plis chronologiques à fenêtre croissante) pour XGBoost — tourner en arrière-plan.
- [ ] Régression : directe et deux modèles (classifieur × régression conditionnelle) → MAE, RMSE.
- [ ] Pendant que ça tourne : **rédiger §4 Méthodologie** (architecture, prétraitement, Algorithme 1 pour l'historique sans fuite, modèles, réglage, rééquilibrage pour Olist).

**Soir**
- [ ] Fin des lectures bloc A ; vérification Crossref des entrées `[A VERIFIER]` du bloc D (classiques).

**Livrable :** tableau principal DataCo (moyenne ± écart-type, deux protocoles, Δ) ; §4 terminée.

## Samedi 17/10 — Olist, analyses, résultats (~10 h)

**Matin**
- [ ] Relancer **exactement le même script** sur Olist (aucun réglage spécifique hors seuil).
- [ ] Profil de dérive : métrique par trimestre de test + prévalence par trimestre, une figure par jeu.

**Après-midi**
- [ ] Analyse de coût C(θ) sur Olist : seuil réglé sur la validation, rapports c_FN/c_FP illustratifs (p. ex. 5:1 et 10:1), C_chr − C_alé.
- [ ] Figures finales dans `reports/figures/` (générées par script).
- [ ] **Rédiger §6** : un sous-titre par question (C1 comparaison des modèles, C2 écart et dérive, RQ1 réplication Olist, RQ2 coût), puis discussion, menaces à la validité (date estimée Olist = promesse commerciale, granularité DataCo, coûts illustratifs, deux jeux seulement).

**Soir**
- [ ] §7 Conclusion (réponse explicite à RQ, RQ1, RQ2 ; limites ; travaux futurs : P2 non faits).
- [ ] Résumé (150–250 mots, au moins un chiffre).
- [ ] Remplir les dernières cases du tableau 1, résoudre ou convertir en question les `\verif` restants.

**Livrable : version FR complète, compilée sans erreur.**

## Dimanche 18/10 — traduction et envoi (~8 h)

**Matin**
- [ ] Relance complète du pipeline sur une copie propre : les chiffres doivent être identiques.
- [ ] Traduction FR → EN section par section dans `supply_chain_resilience.tex` (prose seulement ; équations, tableaux et labels inchangés).

**Après-midi**
- [ ] Fusion des bibliographies en un seul `.bib` ; plus aucun `@misc` pour un article ; compilation EN.
- [ ] Relecture : renvois, numérotation, cohérence des chiffres entre texte, tableaux et résumé.
- [ ] Rédiger le courriel au Prof. Merlec avec les questions (`ORGANISATION.md` §5).

**20 h : envoi** — PDF EN + PDF FR (optionnel) + lien du dépôt si souhaité.

---

## Plan de repli (si une journée glisse)

| Retard constaté | Décision |
|---|---|
| Jeudi soir, Olist pas prêt | Vendredi matin : finir Olist avant les modèles ; RF et régression passent en P1 de samedi. |
| Vendredi soir, Optuna pas terminé | Utiliser les hyperparamètres par défaut + recherche aléatoire courte ; Optuna complet dans la version révisée. |
| Samedi soir, FR incomplète | Envoyer dimanche une **version FR** au Prof. Merlec (avec accord préalable) et traduire après ses retours : la traduction d'un texte qui va changer est du travail perdu. |
| Une référence introuvable | La retirer du texte plutôt que la citer sans l'avoir lue. |

## Suivi

| Indicateur | Mer 14 | Jeu 15 | Ven 16 | Sam 17 | Dim 18 |
|---|---|---|---|---|---|
| `\pending` restants (départ : 7) | | | | 0 | 0 |
| `\tbd` restants (départ : 22) | | | | 0 | 0 |
| `\verif` restants (départ : 12) | | | | | |
| Références `[A VERIFIER]` (départ : 20) | | | | | 0 |
| Sections FR terminées (départ : 2,5 / 7) | | | | 7 / 7 | |
