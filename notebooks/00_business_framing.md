# 00 — Business Understanding

Phase 1 de CRISP-DM. Ce document fixe l'objectif métier, le coût des erreurs et les critères de succès **avant** de toucher aux données. Il alimente le §4.2 de l'article (et le §5.2 pour la règle de coupure).

Les cases « Décision » sont à remplir par moi. Une décision prise ici ne change plus après avoir vu les résultats ; si elle change, je le note dans le journal de bord avec la raison.

---

## 1. Objectif métier

**Question métier :** au moment où une commande est passée, peut-on avertir le gestionnaire qu'elle sera livrée en retard, et de combien, assez tôt pour qu'il agisse ?

| | DataCo | Olist |
|---|---|---|
| Instant de prédiction | Date de commande | Date d'achat |
| Promesse faite au client | Délai prévu, fixé par le mode d'expédition (0, 1, 2 ou 4 jours) | Date de livraison estimée affichée au client |
| Qui exécute la livraison | Transporteur, selon le mode choisi | Vendeur, puis transporteur |
| Leviers possibles après une alerte | Passer à un mode plus rapide, prioriser la préparation, prévenir le client | Relancer le vendeur, prévenir le client, revoir la date affichée |

**Décision — destinataire de l'alerte et action principale retenue dans l'article :**

> Le service client reçoit l'alerte et **prévient le client** du retard en lui annonçant une nouvelle date de livraison. La même action est retenue pour DataCo et Olist, ce qui rend la comparaison entre les deux jeux directe. (Décidé le 19/10/2026.)

## 2. Pourquoi deux tâches

- **Classification** (`y_clf`) : déclencher ou non l'alerte.
- **Régression** (`y_reg`, jours de retard) : choisir l'action selon l'ampleur. Un retard d'un jour ne justifie pas la même réponse qu'un retard d'une semaine.

Sur DataCo, le retard va de 1 à 4 jours seulement : l'apport de la régression y sera limité. Sur Olist, la distribution des retards est à mesurer en phase 2.

**Décision — règle qui combine les deux sorties (exemple : alerte si p ≥ θ, puis action selon le retard prédit) :**

> **Alerte si p ≥ θ**, où p est la probabilité de retard donnée par le classifieur. En cas d'alerte, **la nouvelle date annoncée = date promise + retard prédit** par la régression conditionnelle (entraînée sur les seules commandes en retard, §3.1). La classification décide d'agir, la régression dit quoi annoncer. (Décidé le 19/10/2026.)

## 3. Coût des erreurs

- **Faux négatif** (retard non signalé) : coût c_FN — client mécontent, pénalité, rupture en aval.
- **Faux positif** (fausse alerte) : coût c_FP — action inutile (mode plus cher, relance, message au client).

Aucun des deux jeux ne contient de coûts réels. On raisonne donc sur le rapport **r = c_FN / c_FP**, présenté comme hypothèse illustrative. Le seuil optimal pour des probabilités calibrées est θ* = 1 / (1 + r) (équation `eq:seuil` du §3.7) :

| r = c_FN / c_FP | 1 | 2 | 5 | 10 |
|---|---|---|---|---|
| θ* | 0,50 | 0,33 | 0,17 | 0,09 |

Recommandation : rapporter les résultats pour plusieurs valeurs de r (analyse de sensibilité) plutôt que pour une seule. Un relecteur ne peut pas contester une hypothèse de coût si l'article montre que la conclusion tient pour toute une plage.

**Décision — valeurs de r étudiées, et valeur du scénario principal :**

> r ∈ {1, 2, 5, 10}, en analyse de sensibilité. **Scénario principal : r = 5** — un retard non annoncé (client mécontent, réclamation) coûte nettement plus qu'un message inutile. Le seuil est réglé sur la validation pour chaque r, jamais sur le test. (Décidé le 19/10/2026.)

## 4. Critères de succès (objectifs data mining)

Fixés maintenant pour qu'on ne puisse pas les ajuster après coup.

| Critère | Proposition | Décision |
|---|---|---|
| Métrique principale, classification | PR-AUC (DataCo et Olist), avec F1 et ROC-AUC en complément | Retenu |
| Métrique principale, régression | MAE, avec RMSE en complément | Retenu |
| Un modèle est utile si… | sa PR-AUC dépasse la prévalence du jeu et celle de la régression logistique | Retenu |
| L'écart Δ entre protocoles est un résultat si… | son intervalle sur les graines exclut 0 | Retenu |
| Nombre de graines | 5 | Retenu |
| Règle de coupure chronologique | les 25 % de commandes les plus récentes en test | Retenu. Coupures provisoires (calcul rapide, à recalculer en phase 3) : 12/05/2017 (DataCo), 19/04/2018 (Olist, après exclusion de 2016 et troncature de fin de période, 19/10/2026) |
| Période retenue pour Olist | à partir du 01/01/2017 (2016 : 267 commandes) | Retenu |

## 5. Situation : hypothèses et risques connus

| Constat (vérifié le 14/10/2026) | Conséquence pour l'étude |
|---|---|
| DataCo : prévalence stable autour de 57 % sur 13 trimestres | Écart entre protocoles probablement faible : à rapporter tel quel |
| DataCo : à partir d'octobre 2017, 1 ligne par commande au lieu de 3, et 8 à 20 produits par mois au lieu de 42 à 53 (vérifié le 19/10/2026) | Décalage des covariables sans dérive de prévalence : 8 043 des 15 725 commandes du test chronologique sont dans ce nouveau régime. À décrire au §5.1 |
| DataCo : délai prévu fixé par le mode, First Class toujours en retard | Données probablement générées : menace à la validité |
| Olist : prévalence de 3,4 % à 12,9 % selon le trimestre | Dérive forte : l'écart a le plus de chances d'apparaître ici |
| Olist : la date estimée est une promesse commerciale | Menace à la validité de construit |
| Aucun coût réel observé | Analyse de sensibilité sur r (section 3) |
| Avec la coupure à 25 %, la prévalence d'Olist passe de 7,8 % (entraînement) à 4,0 % (test) après troncature ; DataCo reste à 57 % | Le test Olist subit un décalage de prévalence : à rapporter avec les résultats (§3.5, lien avec la dérive) |
| Olist : les commandes passées juste avant la fin de l'extraction ne figurent que si elles sont déjà livrées ; les commandes encore en cours (1 107 « shipped ») sont exclues | Vérifié en phase 2 : le délai moyen par semaine d'achat passe de 8 jours à 5,0 puis 1,8 sur les deux dernières semaines. **Décidé (19/10/2026) :** fin d'observation = dernière approbation de commande (03/09/2018) ; on exclut les achats des 29 derniers jours (95e centile du délai de livraison), soit les achats à partir du 05/08/2018 (5 209 commandes). Restent 90 994 commandes, prévalence 6,87 % |

## 6. Ce que j'écris dans l'article

**§4.2 Business Understanding** (environ une demi-page) :
1. L'objectif décisionnel et l'instant de prédiction (section 1 ci-dessus).
2. L'articulation des deux tâches (section 2).
3. Le coût des erreurs et le choix du seuil, en renvoyant à l'équation `eq:seuil` du §3.7 sans la répéter (section 3).

**À reporter ailleurs :**
- La règle de coupure et la période Olist → §5.2.
- Les critères de succès → §5.2 et ouverture du §6.
- Les risques de la section 5 → menaces à la validité (§6).

**Fin de phase :** les trois décisions des sections 1 à 3 et la colonne « Décision » de la section 4 sont remplies et le §4.2 est rédigé.
