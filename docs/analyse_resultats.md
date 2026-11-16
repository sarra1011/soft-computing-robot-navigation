# Analyse des résultats — points clés pour le rapport et la soutenance

Ces chiffres viennent de l'exécution de référence de `python run_experiments.py` (graines 0 à 19, Linux, Python 3.13). Les temps et trajectoires sont déterministes et doivent être identiques chez vous. Seuls les **temps de calcul** dépendent de la machine : reprenez ceux de `results/RESULTS.md` après votre propre exécution.

## 1. Protocole et intégrité
- Réglage sur A et B, validation sur C. Deux versions ont été **figées avant** la campagne de test D, E, F :
  - `sys-v1` : le fascicule tel quel ;
  - `sys-v2` : modification motivée **uniquement** par B et C.
- Les deux versions sont évaluées en entier sur D, E et F (1 essai nominal + 20 essais bruités par couple méthode–scénario). Aucun réglage n'a été fait après consultation de ces scénarios. Seule exception, à déclarer : une exécution exploratoire de `sys-v1` sur D à F pendant le développement, qui n'a conduit à aucune modification.
- Bruit : 2 400 triplets pré-générés par graine. Les trois contrôleurs reçoivent exactement la même séquence.

## 2. Séance 2 — moteur flou
- Vérification manuelle : λBas = 0,875, λHaut = 0,125, Sugeno = 0,275, retrouvés exactement.
- Mamdani de l'exercice : 0,3246 (11 points), 0,3335 (101 points), 0,3337 (1001 points). Le calcul « à la main » par sommes explicites et le moteur coïncident à 10⁻⁶ près.
- Comparaison avec scikit-fuzzy sur 300 points (ρ, α) : écart maximal de 3·10⁻³. Il vient de la défuzzification : scikit-fuzzy intègre la courbe affine par morceaux, alors que le fascicule impose la somme discrète.

## 3. Séance 3 — Mamdani vs Sugeno sans obstacle
- Écart maximal entre les surfaces : 0,066 m/s sur v et 0,24 rad/s sur ω. Deux causes :
  - la sortie floue « arrêt » de Mamdani a un centroïde positif (0,021 m/s) : le robot avance lentement en tournant, alors que Sugeno impose exactement 0 ;
  - le centroïde de TG vaut 1,21, contre la constante Sugeno 1,30.
- Scénario A : la référence classique arrive en 19,4 s, les contrôleurs flous en environ 30 s. La règle Proche → L (0,12 m/s) ralentit beaucoup l'approche finale. C'est un effet des **règles**, pas du moteur d'inférence.
- Bornes ±π : α = +π donne TG et α = −π donne TD pour la même direction physique. La discontinuité est sans danger car v = S (rotation sur place), mais le bruit peut faire hésiter le robot quand la cible est exactement derrière lui.
- Finesse de la grille de défuzzification :

  | Points | Erreur max sur ω | Coût par inférence |
  |---|---|---|
  | 151 | 6,2·10⁻³ rad/s | 19,7 µs |
  | 301 | 2,9·10⁻³ rad/s | 19,6 µs |
  | 601 | 1,3·10⁻³ rad/s | 20,8 µs |
  | Sugeno (sans grille) | — | 6,8 µs |

  Le coût est dominé par la surcharge Python et non par la grille, donc la finesse coûte presque rien ici. En C, ce serait différent.

## 4. Séance 4 — l'échec principal et sa correction
- **Défaut de `sys-v1` (observé sur B)** : dès que l'obstacle passe à |β| > π/2, il n'est plus « frontal » et dfront saute de ≈ 0,8 m à 3 m. L'hystérésis 1,0/1,4 m est alors court-circuitée et le mode alterne à chaque pas :
  - 486 commutations pour Sugeno sur B ;
  - ω oscille entre +0,6 et −1,3 rad/s (voir `figures/s4_chattering_v1_vs_v2.png`).
- **Correction (`sys-v2`)** : l'obstacle évité reste suivi pendant l'évitement, ce qui rend dfront continu. Sur B :
  - 10 commutations au lieu de 486 ;
  - Σ|Δω| passe de 784 à 192 ;
  - en contrepartie, la trajectoire est festonnée et plus longue (12,9 m au lieu de 9,0 m).
- **Effet d'arrondi** : dans `sys-v1`, Mamdani contourne B par la droite, car ω vaut −5,8·10⁻¹⁷ pour α = 0. Le cas « face » n'est donc pas tranché par la convention mais par l'arrondi. `sys-v2` traite |β| ≤ 0,05 rad comme « face ».
- **Concavité (scénario de démonstration U)** :
  - les trois méthodes sont déclarées bloquées vers 21 s ;
  - `sys-v1` s'échappe grâce à son broutement, `sys-v2` reste piégé jusqu'à 120 s ;
  - c'est un minimum local typique de la commande réactive. Proposition : un planificateur global (A* sur grille d'occupation, ou graphe de visibilité) qui fournit des points de passage au contrôleur flou.

## 5. Séance 5 — comparaison sur D, E, F
- **Aucune collision** sur les 360 épisodes de test. La garde minimale réelle la plus faible est ≈ 0,39 m dans E, au-dessus de la marge de 0,15 m. Aucun arrêt préventif n'a été déclenché.
- **`sys-v1`**, temps moyen des essais bruités (20/20 arrivées partout) :

  | Scénario | Classique | Mamdani | Sugeno |
  |---|---|---|---|
  | D | 42,9 s | 77,1 s | 89,3 s |
  | E | 45,9 s | **34,2 s** | 55,8 s |
  | F | 30,7 s | 53,4 s | 59,0 s |

  Le classique est le plus rapide grâce à sa vitesse continue v = min(0,6 ; 0,8ρ). Dans E, Mamdani est le plus rapide : sa sortie « S » non nulle lui permet d'avancer en tournant entre les deux obstacles.
- **`sys-v2` dans E, échec nominal par symétrie** : sur l'axe y = 5, les deux obstacles sont exactement symétriques. L'obstacle suivi change, le robot tourne en rond devant le passage, et les trois méthodes atteignent le temps limite. Avec bruit, la symétrie est rompue et les 20 essais arrivent. C'est le seul cas où **le bruit change le classement** (`results/s5_noise_ranking.txt`).
- **Effets à distinguer** :
  - **moteur d'inférence** : les surfaces Mamdani et Sugeno sont proches, avec les écarts liés à « S » et à « TG » ;
  - **partitions et règles** : la règle Proche → L explique la lenteur finale ;
  - **supervision** : la perception frontale de v1 cause le broutement, et la correction v2 introduit la symétrie bloquante dans E.
- **Coût de calcul** (médiane par décision, machine de référence) : classique ≈ 9 µs (Sugeno pour l'évitement), Sugeno ≈ 10 µs, Mamdani ≈ 25 µs.

## 6. Séance 6 — extensions (le fascicule demande d'en présenter UNE)
- **Option A (MLP 32-16)** :
  - erreurs de test (MAE) : 0,003 à 0,004 m/s sur v, 0,013 à 0,020 rad/s sur ω ;
  - en boucle fermée, l'initialisation 1 (MAE ω de 0,020) fait un détour de 11,0 m dans E au lieu de 7,8 m ;
  - l'initialisation 2, qui a pourtant la plus faible erreur, met 82 s en nominal dans E contre 52 s pour Sugeno ;
  - conclusion : **une faible erreur d'imitation ne garantit pas la qualité de navigation**, et le réseau ne fait qu'imiter Sugeno.
- **Option B (algorithme génétique, 12 individus, 10 générations)** :
  - les trois graines convergent vers sv = 1,20, la borne supérieure : il faut aller plus vite ;
  - J passe de 0,846 (paramètres par défaut) à 0,798 ;
  - la recherche aléatoire atteint 0,800 à 0,813 avec le même nombre d'évaluations (≈ 102). Le gain de l'AG est réel mais modeste sur 3 paramètres ;
  - sur le test, l'arrivée est plus rapide (D : 77,1 s au lieu de 89,3 s), mais la garde minimale baisse de 0,80 à 0,68 m ;
  - **le score pondéré échange de la sécurité contre de la vitesse** : il ne constitue pas une contrainte de sécurité.

## 7. Limites
La perception est idéale (carte connue, aucune occlusion) ; la cinématique n'a ni inertie ni modèle moteur ; les obstacles sont uniquement des disques fixes ; il n'y a que 6 cartes. Une supériorité observée ici n'est pas une garantie générale.
