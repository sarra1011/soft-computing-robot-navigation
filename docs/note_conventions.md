# Note de conventions — Séance 1 (simulateur et référence)

**Mini-projet Soft Computing — navigation d'un robot mobile · paramètres `params-v1`**

## 1. Repère, état et commandes
- Arène carrée 10 m × 10 m, origine en bas à gauche, x vers la droite, y vers le haut. Unités SI : mètres, secondes, radians.
- État réel q = (x, y, θ). θ est positif dans le sens trigonométrique et ramené dans [−π, π] par `wrap(a) = atan2(sin a, cos a)`.
- Commande u = (v, ω) : v ∈ [0 ; 0,60] m/s (pas de marche arrière), ω ∈ [−1,50 ; 1,50] rad/s. La saturation est appliquée **par le simulateur**, après le contrôleur, pour tous les contrôleurs.
- Cinématique (Euler explicite, Δt = 0,05 s) : x₊ = x + Δt·v·cos θ, y₊ = y + Δt·v·sin θ, θ₊ = wrap(θ + Δt·ω). Le déplacement d'un pas est donc un segment exact, ce qui permet de tester la collision sur le segment et pas seulement sur ses extrémités.
- Roues : Ωd = (v + bω/2)/r et Ωg = (v − bω/2)/r avec r = 0,05 m, b = 0,30 m. Pour v = 0, les roues tournent en sens opposés (rotation sur place). Aucun modèle moteur n'est imposé.

## 2. Grandeurs géométriques
- ρ = distance robot–cible ; α = wrap(atan2(yg − y, xg − x) − θ). **α > 0 : la cible est à gauche, donc ω > 0 (virage à gauche).**
- Garde réelle d = min_i(‖(x, y) − c_i‖ − r_oi − R), R = 0,20 m. Collision si d ≤ 0 ou si le disque touche/franchit la frontière (distance au mur − R ≤ 0). La garde minimale d'un épisode est calculée sur tous les segments parcourus. Sans obstacle, d = +∞.
- Arrivée : ρ réel ≤ 0,20 m ; on impose alors v = ω = 0 et l'épisode s'arrête. Durée maximale 120 s (2 400 pas).

## 3. Convention temporelle du journal
La ligne k contient l'état **réel** q_k à t_k = k·Δt, l'observation faite à t_k, et la commande u_k appliquée sur [t_k, t_k + Δt[. La dernière ligne est l'état terminal (`arrived`, `collision` ou `timeout`) avec u = (0, 0). Le temps d'arrivée vaut T = K·Δt, K étant l'indice de la première ligne où ρ réel ≤ 0,20 m. Les variations de commande Σ|v_k − v_{k−1}| et Σ|ω_k − ω_{k−1}| sont calculées sur les commandes réellement calculées par le contrôleur (lignes `running`).

## 4. Séparation état réel / observation
Le simulateur conserve l'état réel ; le contrôleur ne reçoit qu'un objet `Observation` (pose estimée, ρ et α recalculés depuis cette pose). Dans le cas nominal la pose observée est exacte ; dans le cas bruité, 2 400 triplets (εx, εy, εθ) sont pré-générés par graine, de sorte que tous les contrôleurs reçoivent la même séquence. Arrivée et collision utilisent toujours l'état réel.

## 5. Loi de référence
v = min(0,60 ; 0,8·ρ)·max(0 ; cos α) ; ω = clip(2α ; −1,50 ; 1,50) ; arrêt si ρ ≤ 0,20 m.
Le facteur cos α annule la vitesse d'avance quand la cible est derrière le robot (|α| ≥ π/2), qui tourne alors sur place avant d'avancer ; sans lui, le robot s'éloigne de la cible pendant le virage (voir `figures/session1_ablation_cos_saturation.png`). Les saturations rendent la commande réalisable : sans elles, v atteint 3,2 m/s et ω 4,7 rad/s, soit 5,4 fois v_max et 3,1 fois ω_max.

## 6. Vérifications réalisées (`python -m unittest discover -s tests -v`)
`wrap` près de ±π ; immobilité (v = ω = 0) ; rotation pure (v = 0) ; ligne droite (ω = 0) ; α > 0 ⇒ virage à gauche ; conversions des roues (aller-retour) ; symétrie gauche–droite d'une trajectoire miroir ; détection de collision sur un segment ; validité des scénarios A–F (ouverture de 1,20 m dans E) ; reproductibilité à configuration et graine identiques ; comparaison Δt = 0,05 s / 0,025 s.

## 7. Questions du fascicule
**La trajectoire la plus courte est-elle nécessairement la plus rapide ?** Non. T = ∫ ds / v : la durée dépend du profil de vitesse autant que de la longueur. Ici v est bornée, réduite par cos α quand le robot n'est pas aligné, proportionnelle à ρ près de la cible, et les virages sur place coûtent du temps (|ω| ≤ 1,50 rad/s). Avec les gains par défaut sur A_back, la variante sans saturations parcourt 6,14 m en 4,85 s alors que la référence parcourt 5,56 m en 10,95 s : un chemin plus long peut être plus rapide. Avec obstacles, le chemin le plus court passe près des disques et impose des ralentissements et des marges.

**Pourquoi la cinématique ne permet-elle pas de conclure sur la consommation électrique réelle ?** Elle ne décrit que la géométrie du mouvement. La consommation dépend des couples (inertie, accélérations, frottements, charge), du modèle électrique des moteurs (résistance, constante de couple, rendement), des pertes de transmission et de la batterie. Ici les commandes changent instantanément (accélération infinie, aucune inertie) et aucune limite ni modèle moteur n'est imposé aux roues : Ωd et Ωg sont calculées mais la puissance mécanique ou électrique est inconnue.
