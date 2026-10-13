# Organisation du travail — analyse et méthode

**Mise à jour :** mercredi 14/10/2026
**Objectif :** version française complète samedi soir, version anglaise envoyée au Prof. Merlec **dimanche 18/10/2026 au plus tard** (voir [`PLANNING.md`](PLANNING.md)).

---

## 1. Diagnostic : où en est le projet

| Bloc | État | Commentaire |
|---|---|---|
| Cadrage (RQ, RQ1, RQ2, contributions) | ✅ Solide | Contribution centrée sur le protocole d'évaluation, pas sur un algorithme : bon angle, défendable. |
| §1 Introduction (FR) | 🟡 ~85 % | Manque une donnée chiffrée sur le coût des retards (avec source) ; un `\verif` sur la nouveauté. |
| §2 Travaux connexes (FR) | 🟡 ~60 % | Structure par angle excellente, mais **la plupart des articles du domaine n'ont pas encore été lus** : tableau 1 rempli de `\tbd` (22 cases), 12 `\verif` dans le texte. |
| §3 Cadre mathématique (FR) | ✅ ~95 % | Complet et rigoureux. Petites coquilles (voir §4 ci-dessous). Un choix à valider avec le Prof. Merlec (architecture à deux modèles). |
| §4 Méthodologie | ❌ 0 % | Peut s'écrire **avant** les résultats (c'est la description de ce qu'on fait). |
| §5 Protocole expérimental | ❌ 0 % | Structure rédigeable tout de suite ; chiffres (effectifs, prévalence, τ*) après l'EDA. |
| §6 Résultats et discussion | ❌ 0 % | Dépend entièrement des expériences. |
| §7 Conclusion, résumé | ❌ 0 % | En dernier. |
| Bibliographie FR | 🟡 | 33 entrées, dont 20 marquées `[A VERIFIER]` ; 4 articles en `@misc` sans revue (+ 2 jeux de données). |
| **Code** | ❌ **0 %** | Le package est encore le gabarit cookiecutter (`dataset.py`, `features.py`, `train.py` = code factice). Données DataCo et Olist téléchargées dans `data/raw/dataco/` et `data/raw/olist/`. |
| Version anglaise | ⏸ Squelette | `supply_chain_resilience.tex` : sections vides. À remplir par traduction à la fin. |
| Gabarit Elsevier / Procedia | 🗑 Supprimé | Cohérent avec le plan : la revue/conférence cible sera décidée avec le Prof. Merlec. |

### Le vrai goulot d'étranglement

**Ce n'est pas la rédaction, ce sont les expériences.** Les sections 1–3 représentent déjà ~50 % du texte final. Les sections 5–7 ne peuvent pas exister sans chiffres, et il n'y a aujourd'hui ni données réelles ni code. Le planning est donc construit autour d'une règle simple :

> **Le code tourne le matin, la rédaction se fait pendant que le code tourne et le soir.**
> Lecture d'articles et vérification bibliographique dans les « trous » (entraînements Optuna, téléchargements).

## 2. Périmètre : ce qui est obligatoire pour dimanche, ce qui peut attendre

Le cadre mathématique promet beaucoup. Pour tenir le délai, on distingue un noyau qui **répond à la RQ** et des extensions.

| Priorité | Élément | Pourquoi |
|---|---|---|
| **P0 — indispensable** | DataCo : partition aléatoire groupée vs chronologique, LogReg + XGBoost, classification | C'est la réponse directe à la RQ. |
| P0 | Olist : même expérience | C'est RQ1 / C3. Sans Olist, la contribution tombe. |
| P0 | Écart Δ sur PR-AUC, F1, ROC-AUC + prévalence par période | Cœur de C2. |
| P0 | Plusieurs graines (≥ 5) → moyenne ± écart-type | Sans variabilité, Δ n'est pas interprétable. |
| **P1 — fortement souhaité** | Forêt aléatoire | Annoncée en C1. |
| P1 | Régression (MAE, RMSE) : directe + architecture à deux modèles | Annoncée dans l'intro et §3.1. |
| P1 | Profil de dérive trimestriel (une figure par jeu) | Annoncé en C2. |
| P1 | Analyse de coût C(θ) sur Olist (coûts illustratifs) | Répond à RQ2. |
| P1 | Optuna (TPE) avec plis chronologiques, budget modeste (50 essais) | Annoncé en C1. |
| **P2 — si le temps le permet** | Comparaison grille / aléatoire / TPE (§3.8) | Intéressant mais hors RQ → **à réduire à une phrase ou à reporter** en « travaux futurs » si besoin. |
| P2 | SHAP, McNemar / Wilcoxon, ablation géographique (piège D7) | Renforce, mais peut être ajouté dans la version révisée. |
| P2 | Comparaison pondération / SMOTE / ajustement du seuil sur Olist | Idem. |

> Si P2 n'est pas fait dimanche, **ne pas le promettre dans le texte** : retirer ou mettre au conditionnel les phrases de §3 qui l'annoncent (notamment « Trois stratégies de recherche sont comparées », §3.8).

## 3. Méthode de travail

### 3.1 Rédaction en français, traduction à la fin

- Toute la rédaction se fait dans `reports/draft/supply_chain_resilience_fr.tex`.
- Les macros `\pending{}` (rouge), `\verif{}` (orange) et `\tbd` restent jusqu'au dernier jour : elles sont la liste de tâches visible dans le PDF. **Objectif dimanche : 0 `\pending`, 0 `\tbd` ; les `\verif` restants sont soit résolus, soit transformés en questions explicites pour le Prof. Merlec.**
- Compter les marqueurs à tout moment :
  ```bash
  grep -c '\\pending\|\\verif\|\\tbd' reports/draft/supply_chain_resilience_fr.tex
  ```
- Les termes anglais clés sont déjà notés en commentaire (mots-clés) : continuer cette pratique pour les termes techniques (`% EN: optimism gap`, `% EN: drift profile`), la traduction sera plus rapide et plus cohérente.
- **Traduction :** dimanche matin, section par section, dans `supply_chain_resilience.tex`, avec la bibliographie fusionnée (voir §3.3). Les équations, tableaux et labels ne changent pas : seule la prose est traduite.

### 3.2 Expériences

- Ordre : DataCo complet (P0) → Olist (P0) → extensions P1.
- Chaque résultat chiffré qui entre dans l'article sort d'un script et est sauvegardé (CSV dans `reports/`, figure dans `reports/figures/`). On ne recopie jamais un chiffre lu dans un notebook.
- Fixer et noter les graines, la date de coupure τ*, les versions de bibliothèques.

### 3.3 Bibliographie

- Liste de recherche : [`references/LISTE_ARTICLES.md`](../references/LISTE_ARTICLES.md).
- Pour chaque article trouvé :
  1. PDF dans `references/` nommé `auteurAnnee_motcle.pdf` (ex. `douaioui2024_late-delivery.pdf`) ;
  2. BibTeX officiel récupéré sur Crossref / l'éditeur, collé dans `related_work_fr.bib` à la place de l'entrée existante (même clé de citation !) ;
  3. champ `note` → `[OK]` ;
  4. cocher la ligne dans `LISTE_ARTICLES.md` et remplir les colonnes du tableau 1 de l'article (données, tâche, modèles, **protocole**, métriques).
- **Fusion des deux .bib** : `related_work.bib` (ancien, EN) contient des entrées absentes de `related_work_fr.bib` (SHAP, SMOTE, Kang & Bhawna, Aoufi & Haloua, Li & Chen, Kasali et al., Wu et al., Pourhejazy et al., Huang). Ne garder à terme **qu'un seul** fichier, avec les clés de la version FR.

### 3.4 Git

Le dépôt n'a qu'un commit et beaucoup de fichiers non suivis. Proposition :
- un commit maintenant (docs + brouillon FR + bib FR) ;
- ensuite un commit par étape du planning (« EDA DataCo », « §4 rédigée »…) ;
- ajouter `.idea/`, `*.synctex.gz` et les PDF compilés au `.gitignore` (`/data/` est déjà ignoré : licences, volume).

## 4. Points relevés dans le brouillon FR (à corriger)

| Où | Problème | Suggestion |
|---|---|---|
| §1, RQ1 | « La dégradation **observée** sous évaluation temporelle… » : observée où ? Le lecteur ne sait pas de quelle observation on parle, et la formulation présuppose le résultat. | Reformuler sans présupposer le résultat (ex. « Un écart entre les deux protocoles apparaît-il… »). |
| §1 | `\pending` : coût chiffré des retards avec source. | Chercher un rapport ou article (voir LISTE_ARTICLES, bloc F). |
| §2.1, §2.5, tableau 1 | Douaioui et al. sur DataCo : affirmé dans le dictionnaire de données, `\verif` dans l'article. | Lire l'article en priorité absolue : si c'est bien DataCo, c'est **le** point de comparaison. |
| §2.2 | Argument « les ensembles d'arbres suffisent sur données tabulaires » non sourcé. | Citer Grinsztajn et al. (2022) et Shwartz-Ziv & Armon (2022) (bloc E de la liste). |
| §3.3, « Types de dérive » | « comme dans l'équation~\eqref{eq:facto} » renvoie à l'équation qui suit immédiatement, et la phrase se termine par une virgule après l'équation. | Supprimer le renvoi ; terminer l'équation par un point. |
| §3.4, Huber | L'équation se termine par une virgule puis une nouvelle phrase commence par une majuscule. | Remplacer la virgule par un point. |
| §3.1 | Architecture à deux modèles (`\verif{valider avec le Prof. Merlec}`). | Question à poser (voir §5). |
| §3.8 | Promet la comparaison grille / aléatoire / TPE. | Garder seulement si fait (voir §2, P2). |
| Toutes sections | SHAP et SMOTE sont dans le plan mais pas cités dans le texte FR. | Les citer en §4 s'ils sont utilisés ; sinon les retirer de la bib. |
| Bib | `\bibliographystyle{plain}` trie par auteur, en anglais. | Acceptable pour la relecture ; le style définitif dépendra de la cible de soumission. |

## 5. Questions à préparer pour le Prof. Merlec

À joindre au courriel d'envoi dimanche (courtes, numérotées) :

1. Validez-vous l'architecture à deux modèles (classifieur + régression conditionnelle) avec la régression directe comme point de comparaison ?
2. La comparaison des stratégies d'optimisation (grille / aléatoire / TPE) doit-elle rester dans l'article, ou se limiter à Optuna ?
3. Quelle cible de soumission (revue / conférence) et donc quel gabarit et quelle limite de pages ?
4. Coûts illustratifs c_FN / c_FP : rapport 5:1 et 10:1 vous paraît-il raisonnable, ou avez-vous une source ?
5. Liste des `\verif` non résolus (s'il en reste).

## 6. Ce que Claude peut / ne doit pas faire

Le plan de rédaction pose que **tout le texte scientifique, le code et l'analyse restent votre travail**. Dans ce cadre, des aides utiles et légitimes :
- relire une section rédigée et pointer incohérences, notations, renvois cassés ;
- vérifier la compilation LaTeX et la cohérence des `\label`/`\ref`/`\cite` ;
- vérifier une entrée BibTeX contre Crossref ;
- relire votre code pour y chercher des fuites (split, encodage cible, historique) ;
- aider à la traduction FR → EN dimanche en relisant votre version anglaise.
