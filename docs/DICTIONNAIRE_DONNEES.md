# Dictionnaire de données — DataCo · Olist

**Projet :** Machine Learning-Based Delivery Delay Prediction for Supply Chain Resilience
**Auteur :** Mupasa Kalunga Jean-Luc — M2 Data Science, UDBL
**Version :** 3.0 — deux jeux réels, publics et indépendants

---

## 0. Statut des deux jeux

| Jeu | Rôle dans l'article | Nature | Volume |
|---|---|---|---|
| **DataCo Smart Supply Chain** | **Jeu principal** — tous les résultats du corps du texte | Réel, public, DOI | 180 519 lignes × 53 colonnes |
| **Olist Brazilian E-Commerce** | **Validation externe** — réplication du résultat principal | Réel, public | ≈ 99 441 commandes, 9 fichiers |

Deux règles qui ne se négocient pas :

1. Le même code de préparation, le même protocole de partitionnement et les mêmes métriques s'appliquent aux deux jeux réels. Si le traitement diffère, la comparaison inter-jeux ne prouve plus rien.
2. L'instant de prédiction est fixé et déclaré : **le moment de la commande**. Toute colonne postérieure à cet instant est une fuite, quel que soit son nom.

---

## 1. DataCo Smart Supply Chain — jeu principal

**Source :** Constante, F., Silva, F., Pereira, A. (2019). *DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS*, Mendeley Data, v5. DOI : 10.17632/8gx2fvg2k6.5 — licence CC BY 4.0.
**Fichiers :** `DataCoSupplyChainDataset.csv` (structuré), `DescriptionDataCoSupplyChain.csv` (descriptif des variables), `tokenized_access_logs.csv` (clickstream, non utilisé ici).
**Lecture :** `pd.read_csv(..., encoding='latin-1')` — l'encodage UTF-8 échoue.
**Période :** commandes de 2015 à début 2018. **Granularité : une ligne = un article de commande**, pas une commande.

C'est le jeu utilisé par Douaioui et al. (2024) et par plusieurs articles de votre bibliographie : vos résultats seront directement comparables, et ce point doit être écrit explicitement dans l'article.

### 1.1 Les 53 colonnes

**Expédition et livraison**

| Colonne | Type | Description | Statut |
|---|---|---|---|
| `Days for shipping (real)` | entier | Délai de livraison effectif | **Source de la cible, puis à retirer de X** |
| `Days for shipment (scheduled)` | entier | Délai contractuel | Utilisable |
| `Delivery Status` | catégoriel | Advance shipping / Shipping on time / Late delivery / Shipping canceled | **FUITE** — sert uniquement à filtrer les annulations |
| `Late_delivery_risk` | binaire | Indicatrice de retard | **FUITE** — reproduit la cible, sert à la vérifier |
| `Shipping Mode` | catégoriel | First Class / Second Class / Standard Class / Same Day | Utilisable |

**Commande**

| Colonne | Type | Remarques |
|---|---|---|
| `Type` | catégoriel | Mode de paiement (DEBIT, TRANSFER, PAYMENT, CASH) |
| `order date (DateOrders)` | date-heure | **Clé du split temporel** — à parser explicitement |
| `shipping date (DateOrders)` | date-heure | Postérieure à la commande → à écarter de X |
| `Order Id` | entier | **Se répète : plusieurs articles par commande** |
| `Order Item Id`, `Order Item Cardprod Id` | entier | Identifiants de ligne |
| `Order Item Quantity` | entier | Quantité |
| `Order Item Discount`, `Order Item Discount Rate` | numérique | Remise |
| `Order Item Product Price`, `Order Item Total`, `Sales` | numérique | Montants |
| `Order Item Profit Ratio`, `Order Profit Per Order`, `Benefit per order`, `Sales per customer` | numérique | Marges — postérieures à la vente, à discuter |
| `Order Status` | catégoriel | COMPLETE, PENDING, CANCELED, SUSPECTED_FRAUD… | **FUITE** — état final |

**Client, géographie, produit**

| Colonne | Type | Remarques |
|---|---|---|
| `Customer Id`, `Order Customer Id` | entier | Identifiants |
| `Customer Segment` | catégoriel | Consumer / Corporate / Home Office |
| `Customer City`, `Customer State`, `Customer Country`, `Customer Zipcode` | catégoriel | 1 valeur manquante sur Zipcode |
| `Market`, `Order Region`, `Order Country`, `Order City`, `Order State` | catégoriel | Destination — forte cardinalité sur City (≈ 3 600) et State (≈ 1 100) |
| `Latitude`, `Longitude` | numérique | Coordonnées client |
| `Category Id`, `Category Name`, `Department Id`, `Department Name` | catégoriel | Taxonomie produit |
| `Product Card Id`, `Product Category Id`, `Product Name`, `Product Price` | mixte | Produit |
| `Order Zipcode` | numérique | **≈ 86 % manquant** |
| `Product Status` | entier | Constante (0) → sans information |
| `Product Description` | — | **Vide à 100 %** → à supprimer |
| `Product Image` | texte | URL → à supprimer |
| `Customer Email`, `Customer Password`, `Customer Street`, `Customer Fname`, `Customer Lname` | texte | **Données personnelles → à retirer, et à déclarer retirées dans l'article** |

### 1.2 Construction des cibles

```python
delta  = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
y_clf  = (delta > 0).astype(int)      # classification : retard oui/non
y_reg  = delta.clip(lower=0)          # régression : jours de retard
```

Vérification obligatoire : `y_clf` doit coïncider avec `Late_delivery_risk`. Si l'accord n'est pas quasi parfait, votre parsing est faux — corrigez-le avant d'aller plus loin, et mentionnez cette vérification dans l'article (c'est court et cela inspire confiance).

**Prévalence : environ 55 % de retards.** Conséquence directe : *il n'y a pas de déséquilibre de classes sur DataCo*. L'argumentaire PR-AUC, SMOTE et seuil optimisé garde tout son sens sur Olist, pas ici. Le dire explicitement est un bon point, pas un aveu de faiblesse : cela montre que vous adaptez le protocole aux données plutôt que d'appliquer une recette.

### 1.3 Colonnes à retirer de X

| Motif | Colonnes |
|---|---|
| Fuite (postérieur à l'expédition) | `Days for shipping (real)`, `Delivery Status`, `Late_delivery_risk`, `Order Status`, `shipping date (DateOrders)` |
| Vide ou constante | `Product Description`, `Product Status` |
| Données personnelles | `Customer Email`, `Customer Password`, `Customer Street`, `Customer Fname`, `Customer Lname` |
| Bruit / identifiants sans valeur prédictive | `Product Image`, `Order Item Id`, `Order Item Cardprod Id` |
| À discuter | `Order Zipcode` (86 % manquant), marges et profits (postérieurs à la vente) |

### 1.4 Les pièges de ce jeu

- **D1 — Granularité.** Une ligne = un article. Un `Order Id` peut apparaître plusieurs fois avec la même date, le même client, le même délai. Un split aléatoire ligne à ligne place alors des morceaux de la même commande des deux côtés : **fuite train → test**. Deux solutions défendables : agréger au niveau commande (quantité totale, montant total, nombre d'articles, catégorie dominante) ou faire un `GroupShuffleSplit` / `GroupKFold` par `Order Id`. Choisissez, justifiez, et mentionnez l'autre option en limite.
- **D2 — Fuite évidente.** Garder `Late_delivery_risk` ou `Days for shipping (real)` donne une AUC de 1,00. C'est le test de bon fonctionnement de votre vigilance, pas un résultat.
- **D3 — Dates.** `order date (DateOrders)` et `shipping date (DateOrders)` sont des chaînes ; parsez-les explicitement avec un format déclaré, jamais en laissant pandas deviner.
- **D4 — Cardinalité.** `Order City`, `Order State`, `Product Name` ont des centaines à des milliers de modalités. One-hot les fait exploser ; l'encodage ordinal introduit un ordre factice. Encodage cible avec ajustement **à l'intérieur du pipeline sur le train uniquement**, ou regroupement des modalités rares.
- **D5 — Doublons.** Vérifiez les lignes strictement identiques après suppression des identifiants ; dédoublonnez **avant** le split.
- **D6 — Statuts anormaux.** `Order Status` contient `SUSPECTED_FRAUD` et `CANCELED`. Décidez si vous les gardez, comptez-les, déclarez la décision.
- **D7 — Géographie peu informative.** Plusieurs travaux constatent que la région n'explique presque rien du retard dans ce jeu. Ne le supposez pas : montrez-le, éventuellement par une ablation. C'est une ligne de tableau gratuite.

---

## 2. Olist Brazilian E-Commerce — validation externe

**Source :** Olist, *Brazilian E-Commerce Public Dataset by Olist*, Kaggle — licence CC BY-NC-SA 4.0. Données commerciales réelles, anonymisées.
**Volume :** ≈ 99 441 commandes, 2016 → 2018, réparties sur 9 fichiers CSV.
**Granularité :** `olist_orders_dataset` est au niveau **commande** ; `olist_order_items_dataset` au niveau **article**.

### 2.1 Les fichiers et leurs jointures

| Fichier | Clé | Contenu utile |
|---|---|---|
| `olist_orders_dataset.csv` | `order_id`, `customer_id` | `order_status`, `order_purchase_timestamp`, `order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date`, `order_estimated_delivery_date` |
| `olist_order_items_dataset.csv` | `order_id` + `order_item_id` | `product_id`, `seller_id`, `shipping_limit_date`, `price`, `freight_value` |
| `olist_products_dataset.csv` | `product_id` | poids, dimensions, catégorie, nombre de photos |
| `olist_customers_dataset.csv` | `customer_id` | `customer_unique_id`, `customer_zip_code_prefix`, ville, État |
| `olist_sellers_dataset.csv` | `seller_id` | code postal, ville, État du vendeur |
| `olist_geolocation_dataset.csv` | `zip_code_prefix` | latitude, longitude → **distance client-vendeur** |
| `olist_order_payments_dataset.csv` | `order_id` | type de paiement, mensualités, montant |
| `olist_order_reviews_dataset.csv` | `order_id` | **postérieur à la livraison → fuite** (réservé à une éventuelle piste NLP hors cible) |
| `product_category_name_translation.csv` | catégorie | traduction PT → EN |

Chaîne de jointure recommandée : `orders` ← `order_items` (agrégés par commande) ← `products` ; puis `customers` et `sellers` ; puis `geolocation` (moyenne des coordonnées par préfixe de code postal, car le fichier contient plusieurs lignes par préfixe).

### 2.2 Construction des cibles

```python
delta = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
y_clf = (delta > 0).astype(int)
y_reg = delta.clip(lower=0)
```

**Filtres, à compter et à déclarer :** ne conservez que `order_status == "delivered"`, puis écartez les lignes où `order_delivered_customer_date` est vide. Donnez les deux volumes exacts dans l'*Experimental Setup*.

**Prévalence : faible** — une petite minorité de commandes dépassent la date estimée. Mesurez-la vous-même et rapportez-la. C'est ici, et seulement ici, que le déséquilibre de classes, la PR-AUC, l'optimisation du seuil et le traitement du rééquilibrage deviennent indispensables.

### 2.3 Colonnes disponibles à l'instant de la commande

| Utilisable | Non utilisable (postérieur à l'achat) |
|---|---|
| `order_purchase_timestamp` (et ses dérivés calendaires) | `order_approved_at` |
| `order_estimated_delivery_date` et le délai promis | `order_delivered_carrier_date` |
| `price`, `freight_value`, ratio fret / prix | `order_delivered_customer_date` (cible) |
| poids, volume, catégorie, nombre d'articles, nombre de vendeurs | `order_status` |
| États et codes postaux client / vendeur, distance calculée | avis clients (`review_*`) |

`shipping_limit_date` est fixée à la commande : elle est utilisable, mais dites-le, car un relecteur se posera la question.

### 2.4 Caractéristiques à construire

Distance client-vendeur (haversine sur les coordonnées moyennes par préfixe), délai promis en jours, même État ou non, nombre d'articles, nombre de vendeurs distincts, poids total, ratio fret / prix, variables calendaires (mois, jour de semaine, heure, week-end), et historique de ponctualité du vendeur **en fenêtre strictement antérieure à la commande courante**. Cette dernière est la plus délicate : c'est elle qui mérite l'Algorithme 1 en pseudo-code dans l'article.

### 2.5 Les pièges de ce jeu

- Dates manquantes en cascade sur les commandes non livrées.
- `olist_geolocation_dataset` a plusieurs lignes par code postal : agrégez avant de joindre, sinon la jointure duplique les commandes.
- Une commande peut avoir plusieurs vendeurs : décidez d'une règle (vendeur principal ? le plus éloigné ?) et justifiez-la.
- La date estimée est une promesse commerciale, pas un engagement contractuel homogène : c'est une menace à la validité de construit, à écrire dans la section *Threats to Validity*.
- Licence non commerciale (CC BY-NC-SA) : citez-la.

---

## 3. Correspondance entre les deux jeux

C'est ce tableau qui rend la comparaison inter-jeux lisible. Il a sa place dans l'*Experimental Setup*.

| Concept | DataCo | Olist |
|---|---|---|
| Instant de prédiction | `order date (DateOrders)` | `order_purchase_timestamp` |
| Référence contractuelle | `Days for shipment (scheduled)` | `order_estimated_delivery_date` |
| Réalisation | `Days for shipping (real)` | `order_delivered_customer_date` |
| Cible classification | `delta > 0` | `delta > 0` |
| Prévalence | ≈ 55 % | faible (à mesurer) |
| Granularité brute | article de commande | commande + articles |
| Mode de transport | `Shipping Mode` (4 modalités) | absent |
| Distance | absente (lat/lng client seulement) | calculable (client ↔ vendeur) |
| Segment client | `Customer Segment` | absent |
| Montant | `Sales`, `Order Item Total` | `price` + `freight_value` |
| Géographie | `Market`, `Order Region`, `Order Country` | États brésiliens |
| Acteur de l'exécution | transporteur implicite | `seller_id` |

Ce que ce tableau montre — et qu'il faut commenter dans l'article : les deux jeux ne partagent que le squelette (date de commande, promesse, réalisation). Si l'effet du protocole d'évaluation s'observe malgré des variables explicatives aussi différentes, la conclusion est robuste. C'est l'argument de la contribution C3.

---

## 4. Ordre de travail (CRISP-DM, deux jeux)

1. **Data Understanding.** EDA séparée sur DataCo et sur Olist. Un tableau d'anomalies par jeu. Prévalence, période, granularité, taux de manquants.
2. **Data Preparation.** Un module `preprocessing.py` avec des fonctions testables et un point d'entrée par jeu (`load_dataco`, `load_olist`) débouchant sur le **même schéma de sortie** : `X`, `y_clf`, `y_reg`, `timestamp`, `group_id`. C'est ce schéma commun qui rend la comparaison possible.
3. **Modeling.** Baseline linéaire régularisée → Random Forest → XGBoost → Optuna, à l'intérieur d'un `Pipeline` scikit-learn ajusté sur le train seul, identique sur les deux jeux.
4. **Evaluation.** Split chronologique et `TimeSeriesSplit` ; puis le même protocole en split aléatoire, pour mesurer l'écart. Répétitions multi-graines, intervalles de confiance, McNemar, Wilcoxon. SHAP sur les deux jeux.
5. **Deployment.** Coût du faux négatif contre le faux positif, seuil optimal, ce que le gestionnaire fait de la prédiction. Le seuil se règle sur Olist (déséquilibré) ; sur DataCo, la question est différente et il faut le dire.
