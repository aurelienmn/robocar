import csv
import os
import glob
import numpy as np
import pygame
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

SIMULATOR_PATH = r"C:\Projet\robocar\simulator\RacingSimulator.exe"
CONFIG_PATH = r"C:\Projet\robocar\config\agents.json"
DATA_DIR = r"C:\Projet\robocar\data"

os.makedirs(DATA_DIR, exist_ok=True)

def get_next_lap_number():
    """Trouve le prochain numéro de lap disponible"""
    existing_laps = glob.glob(os.path.join(DATA_DIR, "lap_*.csv"))
    if not existing_laps:
        return 1
    
    numbers = []
    for lap_file in existing_laps:
        basename = os.path.basename(lap_file)
        try:
            # Extraire le numéro de lap_XXXX.csv
            num = int(basename.replace("lap_", "").replace(".csv", ""))
            numbers.append(num)
        except ValueError:
            continue
    
    return max(numbers) + 1 if numbers else 1

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise Exception("Aucune manette Xbox détectée. Branche la manette avant de lancer le script.")

joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Manette détectée : {joystick.get_name()}")

env = UnityEnvironment(
    file_name=SIMULATOR_PATH,
    base_port=5004,
    additional_args=["--config-path", CONFIG_PATH],
    no_graphics=False
)

header = [f"ray_{i}" for i in range(50)] + ["throttle", "steering"]

def clamp(value, min_value=-1.0, max_value=1.0):
    return max(min_value, min(max_value, value))

def apply_deadzone(value, deadzone=0.08):
    if abs(value) < deadzone:
        return 0.0

    # Re-normalise après zone morte pour garder toute la plage [-1, 1]
    if value > 0:
        return (value - deadzone) / (1.0 - deadzone)
    else:
        return (value + deadzone) / (1.0 - deadzone)

def steering_curve(value, gamma=1.8):
    """
    Courbe progressive :
    - plus douce autour du centre
    - plus forte en fin de course
    gamma > 1 = centre plus doux
    """
    return np.sign(value) * (abs(value) ** gamma)

def smooth_value(current, target, alpha=0.18):
    """
    Lissage simple :
    alpha petit = plus fluide
    alpha grand = plus réactif
    """
    return current + alpha * (target - current)

try:
    env.reset()
    print("Connexion OK")
    print("Contrôles manette Xbox :")
    print("- Stick gauche : direction")
    print("- RT : avancer")
    print("- LT : reculer")
    print("- Bouton A : SAUVEGARDER le tour")
    print("- Bouton B : ANNULER et recommencer")
    print("- Bouton Menu/Start : quitter")

    behavior_name = list(env.behavior_specs.keys())[0]

    # Valeurs lissées
    smoothed_steering = 0.0
    smoothed_throttle = 0.0
    
    # Données temporaires du tour en cours
    lap_data = []
    button_a_was_pressed = False
    button_b_was_pressed = False
    
    print(f"📍 Tour en cours - {len(lap_data)} points enregistrés")

    while True:
        pygame.event.pump()

        decision_steps, _ = env.get_steps(behavior_name)

        if len(decision_steps) == 0:
            env.step()
            continue

        obs = decision_steps.obs[0][0]

        # -------------------------
        # Direction améliorée
        # -------------------------
        raw_steering = joystick.get_axis(0)

        # 1. zone morte
        steering = apply_deadzone(raw_steering, deadzone=0.07)

        # 2. courbe progressive
        steering = steering_curve(steering, gamma=1.8)

        # 3. lissage
        smoothed_steering = smooth_value(smoothed_steering, steering, alpha=0.20)

        # -------------------------
        # Accélération / frein
        # -------------------------
        lt_raw = joystick.get_axis(4)
        rt_raw = joystick.get_axis(5)

        lt = (lt_raw + 1) / 2
        rt = (rt_raw + 1) / 2

        throttle = rt - lt
        throttle = apply_deadzone(throttle, deadzone=0.05)
        smoothed_throttle = smooth_value(smoothed_throttle, throttle, alpha=0.25)

        # Clamp final
        smoothed_steering = clamp(smoothed_steering)
        smoothed_throttle = clamp(smoothed_throttle)

        # Bouton A : Sauvegarder le tour
        button_a_pressed = joystick.get_button(0) if joystick.get_numbuttons() > 0 else False
        if button_a_pressed and not button_a_was_pressed:
            if len(lap_data) > 0:
                # Créer un nouveau fichier lap_XXXX.csv
                lap_num = get_next_lap_number()
                lap_filename = os.path.join(DATA_DIR, f"lap_{lap_num:04d}.csv")
                
                with open(lap_filename, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    for row in lap_data:
                        writer.writerow(row)
                
                print(f"✅ Tour sauvegardé : {os.path.basename(lap_filename)} ({len(lap_data)} points)")
                lap_data = []  # Recommencer un nouveau tour
            else:
                print("⚠️  Aucune donnée à sauvegarder")
        button_a_was_pressed = button_a_pressed

        # Bouton B : Annuler et recommencer
        button_b_pressed = joystick.get_button(1) if joystick.get_numbuttons() > 1 else False
        if button_b_pressed and not button_b_was_pressed:
            if len(lap_data) > 0:
                print(f"🔄 Tour annulé ({len(lap_data)} points supprimés)")
                lap_data = []
            else:
                print("ℹ️  Aucun tour en cours")
        button_b_was_pressed = button_b_pressed

        # Quitter avec Menu / Start
        quit_pressed = False
        for btn in [6, 7]:
            if btn < joystick.get_numbuttons() and joystick.get_button(btn):
                quit_pressed = True
                break

        if quit_pressed:
            break

        actions = np.array([[smoothed_throttle, smoothed_steering]], dtype=np.float32)
        env.set_actions(behavior_name, ActionTuple(continuous=actions))

        # Accumuler les données du tour en cours
        row = list(obs) + [smoothed_throttle, smoothed_steering]
        lap_data.append(row)

        env.step()

finally:
    env.close()
    pygame.quit()
    print("Environnement fermé")