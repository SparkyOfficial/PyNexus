"""
Модуль для обчислювальної хімії в PyNexus.
Включає функції для моделювання молекул, розрахунків електронної структури,
квантово-хімічних обчислень, молекулярної динаміки та інших методів обчислювальної хімії.

Автор: Андрій Будильников
"""

import math
import numpy as np
from typing import List, Tuple, Callable, Union, Optional, Dict
from scipy import constants, optimize, linalg
import matplotlib.pyplot as plt

# Молекулярна геометрія та структура
def molecular_geometry_optimizer(atom_positions: List[List[float]], 
                               atom_types: List[str], 
                               method: str = "uff", 
                               max_iterations: int = 1000) -> Tuple[List[List[float]], float]:
    """
    Оптимізація геометрії молекули.
    
    Параметри:
        atom_positions: Початкові позиції атомів [[x1,y1,z1], [x2,y2,z2], ...]
        atom_types: Типи атомів ["C", "H", "O", ...]
        method: Метод оптимізації ("uff", "mmff", "semiempirical")
        max_iterations: Максимальна кількість ітерацій
    
    Повертає:
        Кортеж (optimized_positions, final_energy) з оптимізованими позиціями та енергією
    """
    # Потенціальна енергія молекули (спрощена модель)
    def potential_energy(positions):
        energy = 0.0
        n_atoms = len(positions)
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                # Відстань між атомами
                r = math.sqrt(sum((positions[i][k] - positions[j][k])**2 for k in range(3)))
                
                # Потенціал Леннард-Джонса
                sigma = 3.0  # параметр σ (Ангстрем)
                epsilon = 0.1  # параметр ε (ккал/моль)
                
                if r > 0:
                    lj_energy = 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
                    energy += lj_energy
                
                # Кулонівська взаємодія (для заряджених атомів)
                charges = {"C": 0.0, "H": 0.0, "O": -0.8, "N": -0.8, "Cl": -1.0}
                charge_i = charges.get(atom_types[i], 0.0)
                charge_j = charges.get(atom_types[j], 0.0)
                
                if charge_i != 0 and charge_j != 0 and r > 0:
                    coulomb_energy = 332.0 * charge_i * charge_j / r  # ккал/моль
                    energy += coulomb_energy
        
        return energy
    
    # Градієнт потенціальної енергії
    def energy_gradient(positions):
        gradients = []
        n_atoms = len(positions)
        delta = 1e-6
        
        for i in range(n_atoms):
            grad_atom = []
            for coord in range(3):  # x, y, z
                # Чисельна похідна
                pos_plus = [row[:] for row in positions]
                pos_minus = [row[:] for row in positions]
                pos_plus[i][coord] += delta
                pos_minus[i][coord] -= delta
                
                grad = (potential_energy(pos_plus) - potential_energy(pos_minus)) / (2 * delta)
                grad_atom.append(grad)
            gradients.append(grad_atom)
        
        return gradients
    
    # Оптимізація геометрії методом спуску градієнта
    positions = [row[:] for row in atom_positions]  # копія
    learning_rate = 0.01
    tolerance = 1e-4
    
    for iteration in range(max_iterations):
        energy = potential_energy(positions)
        gradients = energy_gradient(positions)
        
        # Перевірка збіжності
        grad_norm = math.sqrt(sum(sum(g**2 for g in grad_atom) for grad_atom in gradients))
        if grad_norm < tolerance:
            break
        
        # Оновлення позицій
        for i in range(len(positions)):
            for j in range(3):
                positions[i][j] -= learning_rate * gradients[i][j]
    
    final_energy = potential_energy(positions)
    return (positions, final_energy)

def bond_analysis(atom_positions: List[List[float]], 
                 atom_types: List[str]) -> Dict[str, List]:
    """
    Аналіз хімічних зв'язків у молекулі.
    
    Параметри:
        atom_positions: Позиції атомів
        atom_types: Типи атомів
    
    Повертає:
        Словник з інформацією про зв'язки
    """
    bonds = []
    angles = []
    torsions = []
    
    n_atoms = len(atom_positions)
    
    # Ковалентні радіуси атомів (Ангстрем)
    covalent_radii = {
        "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, 
        "F": 0.57, "P": 1.07, "S": 1.05, "Cl": 1.02,
        "Br": 1.20, "I": 1.39
    }
    
    # Визначення зв'язків
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            # Відстань між атомами
            r = math.sqrt(sum((atom_positions[i][k] - atom_positions[j][k])**2 for k in range(3)))
            
            # Сума ковалентних радіусів
            radius_sum = covalent_radii.get(atom_types[i], 0.76) + covalent_radii.get(atom_types[j], 0.76)
            
            # Якщо відстань менша за 1.3 суми радіусів - зв'язок
            if r < 1.3 * radius_sum:
                bond_length = r
                bond_type = atom_types[i] + "-" + atom_types[j]
                bonds.append({
                    "atoms": (i, j),
                    "type": bond_type,
                    "length": bond_length
                })
    
    # Визначення валентних кутів
    for i in range(len(bonds)):
        for j in range(i+1, len(bonds)):
            # Спільний атом
            atom1_i, atom1_j = bonds[i]["atoms"]
            atom2_i, atom2_j = bonds[j]["atoms"]
            
            common_atom = None
            other_atoms = []
            
            if atom1_i == atom2_i:
                common_atom = atom1_i
                other_atoms = [atom1_j, atom2_j]
            elif atom1_i == atom2_j:
                common_atom = atom1_i
                other_atoms = [atom1_j, atom2_i]
            elif atom1_j == atom2_i:
                common_atom = atom1_j
                other_atoms = [atom1_i, atom2_j]
            elif atom1_j == atom2_j:
                common_atom = atom1_j
                other_atoms = [atom1_i, atom2_i]
            
            if common_atom is not None and len(other_atoms) == 2:
                # Обчислення валентного кута
                pos_common = np.array(atom_positions[common_atom])
                pos1 = np.array(atom_positions[other_atoms[0]])
                pos2 = np.array(atom_positions[other_atoms[1]])
                
                vec1 = pos1 - pos_common
                vec2 = pos2 - pos_common
                
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = math.acos(np.clip(cos_angle, -1.0, 1.0)) * 180 / math.pi
                
                angles.append({
                    "atoms": (other_atoms[0], common_atom, other_atoms[1]),
                    "angle": angle
                })
    
    return {
        "bonds": bonds,
        "angles": angles,
        "torsions": torsions
    }

# Електронна структура
def hartree_fock_solver(basis_functions: List[Callable], 
                       nuclear_charges: List[float], 
                       nuclear_positions: List[List[float]], 
                       max_iterations: int = 50, 
                       convergence_threshold: float = 1e-6) -> Dict:
    """
    Розв'язувач рівняння Хартрі-Фока.
    
    Параметри:
        basis_functions: Базисні функції
        nuclear_charges: Заряди ядер
        nuclear_positions: Позиції ядер
        max_iterations: Максимальна кількість ітерацій
        convergence_threshold: Поріг збіжності
    
    Повертає:
        Словник з результатами розрахунку
    """
    # Кількість базисних функцій
    n_basis = len(basis_functions)
    
    # Матриця перекривання
    def overlap_integral(basis_i, basis_j):
        # Спрощений розрахунок інтеграла перекривання
        # Для гаусових функцій: S_ij = ∫ φ_i(r) φ_j(r) dr
        return 1.0 if basis_i == basis_j else 0.1  # приблизне значення
    
    S = np.zeros((n_basis, n_basis))
    for i in range(n_basis):
        for j in range(n_basis):
            S[i, j] = overlap_integral(basis_functions[i], basis_functions[j])
    
    # Гамільтоніан ядра
    def core_hamiltonian_integral(basis_i, basis_j, nuclear_charges, nuclear_positions):
        # Спрощений розрахунок: H_core_ij = ∫ φ_i(r) [-½∇² - Σₐ Zₐ/|r-Rₐ|] φ_j(r) dr
        kinetic = 1.0 if basis_i == basis_j else 0.0  # кінетична енергія
        nuclear_attraction = -sum(Z / 2.0 for Z in nuclear_charges)  # приблизна потенціальна енергія
        return kinetic + nuclear_attraction
    
    H_core = np.zeros((n_basis, n_basis))
    for i in range(n_basis):
        for j in range(n_basis):
            H_core[i, j] = core_hamiltonian_integral(
                basis_functions[i], basis_functions[j], nuclear_charges, nuclear_positions)
    
    # Двочастинкові інтеграли (кулонівські та обмінні)
    def two_electron_integral(i, j, k, l):
        # Спрощений розрахунок двочастинкових інтегралів
        # (ij|kl) = ∫∫ φ_i(r₁) φ_j(r₁) 1/|r₁-r₂| φ_k(r₂) φ_l(r₂) dr₁ dr₂
        if i == j == k == l:
            return 1.0
        elif (i == k and j == l) or (i == l and j == k):
            return 0.5
        else:
            return 0.1
    
    # Початкова густина (нульова)
    P = np.zeros((n_basis, n_basis))
    
    # Основний цикл Хартрі-Фока
    electronic_energy = 0.0
    nuclear_repulsion = 0.0
    
    # Обчислення енергії відштовхування ядер
    for i in range(len(nuclear_charges)):
        for j in range(i+1, len(nuclear_charges)):
            r = math.sqrt(sum((nuclear_positions[i][k] - nuclear_positions[j][k])**2 for k in range(3)))
            if r > 0:
                nuclear_repulsion += nuclear_charges[i] * nuclear_charges[j] / r
    
    energies = []
    
    for iteration in range(max_iterations):
        # Матриця Фока
        F = H_core.copy()
        
        # Додавання кулонівських та обмінних членів
        for i in range(n_basis):
            for j in range(n_basis):
                for k in range(n_basis):
                    for l in range(n_basis):
                        coulomb = P[k, l] * two_electron_integral(i, j, k, l)
                        exchange = -0.5 * P[k, l] * two_electron_integral(i, l, k, j)
                        F[i, j] += coulomb + exchange
        
        # Розв'язання узагальненої задачі власних значень: F C = S C E
        # Метод Хартрі-Фока-Рутаан
        try:
            # Перетворення Левінсона
            S_sqrt = linalg.sqrtm(S)
            S_inv_sqrt = linalg.inv(S_sqrt)
            
            # Узагальнена задача власних значень
            F_prime = S_inv_sqrt @ F @ S_inv_sqrt
            eigenvalues, C_prime = linalg.eigh(F_prime)
            
            # Повернення до початкового базису
            C = S_inv_sqrt @ C_prime
        except:
            # Якщо є проблеми з діагоналізацією, використовуємо просту діагоналізацію
            eigenvalues, C = linalg.eigh(F)
        
        # Нова матриця густини
        P_old = P.copy()
        P = np.zeros((n_basis, n_basis))
        n_electrons = sum(nuclear_charges)  # приблизна кількість електронів
        n_occupied = int(n_electrons / 2)  # заповнені орбіталі
        
        for i in range(n_basis):
            for j in range(n_basis):
                for a in range(n_occupied):
                    P[i, j] += 2 * C[i, a] * C[j, a]  # множник 2 через спін
        
        # Обчислення електронної енергії
        electronic_energy = 0.0
        for i in range(n_basis):
            for j in range(n_basis):
                electronic_energy += P[i, j] * H_core[i, j]
                for k in range(n_basis):
                    for l in range(n_basis):
                        electronic_energy += 0.5 * P[i, j] * P[k, l] * two_electron_integral(i, j, k, l)
                        electronic_energy -= 0.25 * P[i, k] * P[j, l] * two_electron_integral(i, l, k, j)
        
        total_energy = electronic_energy + nuclear_repulsion
        energies.append(total_energy)
        
        # Перевірка збіжності
        if iteration > 0:
            delta_energy = abs(energies[-1] - energies[-2])
            if delta_energy < convergence_threshold:
                break
    
    # Орбітальні енергії
    orbital_energies = eigenvalues.tolist()
    
    # Заповнення орбіталей
    occupation_numbers = [2.0 if i < n_occupied else 0.0 for i in range(n_basis)]
    
    return {
        "total_energy": total_energy,
        "electronic_energy": electronic_energy,
        "nuclear_repulsion": nuclear_repulsion,
        "orbital_energies": orbital_energies,
        "occupation_numbers": occupation_numbers,
        "converged": iteration < max_iterations - 1,
        "iterations": iteration + 1,
        "density_matrix": P.tolist()
    }

def molecular_orbitals_analysis(coefficients: List[List[float]], 
                               orbital_energies: List[float], 
                               atom_positions: List[List[float]], 
                               basis_functions: List[Callable]) -> Dict:
    """
    Аналіз молекулярних орбіталей.
    
    Параметри:
        coefficients: Коефіцієнти розкладання молекулярних орбіталей
        orbital_energies: Енергії орбіталей
        atom_positions: Позиції атомів
        basis_functions: Базисні функції
    
    Повертає:
        Словник з аналізом орбіталей
    """
    n_orbitals = len(orbital_energies)
    n_atoms = len(atom_positions)
    
    # Аналіз участі атомів у молекулярних орбіталях
    atom_contributions = []
    
    for i in range(n_orbitals):
        contributions = []
        for j in range(n_atoms):
            # Приблизний внесок j-го атома у i-ту орбіталь
            # Сума квадратів коефіцієнтів для базисних функцій цього атома
            atom_contribution = 0.0
            # Припускаємо, що кожен атом має одну базисну функцію
            if j < len(coefficients):
                atom_contribution = coefficients[i][j]**2
            contributions.append(atom_contribution)
        
        # Нормалізація
        total_contribution = sum(contributions)
        if total_contribution > 0:
            contributions = [c / total_contribution for c in contributions]
        
        atom_contributions.append(contributions)
    
    # Визначення типу орбіталей (σ, π, n тощо)
    orbital_types = []
    for i in range(n_orbitals):
        # Спрощена класифікація
        if orbital_energies[i] < -10:
            orbital_type = "core"
        elif orbital_energies[i] < 0:
            orbital_type = "bonding"
        elif orbital_energies[i] < 5:
            orbital_type = "non-bonding"
        else:
            orbital_type = "anti-bonding"
        orbital_types.append(orbital_type)
    
    return {
        "atom_contributions": atom_contributions,
        "orbital_types": orbital_types,
        "homo_lumo_gap": orbital_energies[n_orbitals//2] - orbital_energies[n_orbitals//2-1] if n_orbitals > 1 else 0.0
    }

# Квантово-хімічні методи
def density_functional_theory(density: Callable[[float, float, float], float], 
                            external_potential: Callable[[float, float, float], float], 
                            grid_points: List[List[float]], 
                            max_iterations: int = 100) -> Dict:
    """
    Проста реалізація методу функціонала густини (DFT).
    
    Параметри:
        density: Функція електронної густини ρ(r)
        external_potential: Зовнішній потенціал V_ext(r)
        grid_points: Точки сітки для інтегрування
        max_iterations: Максимальна кількість ітерацій
    
    Повертає:
        Словник з результатами DFT
    """
    # Обмінно-кореляційний функціонал (спрощений)
    def exchange_correlation_energy(density_value):
        # Локальний спіновий функціонал (LSDA)
        if density_value > 0:
            return -0.7386 * density_value**(4/3)  # приблизна форма
        return 0.0
    
    def exchange_correlation_potential(density_value):
        # Похідна обмінно-кореляційної енергії
        if density_value > 0:
            return -0.7386 * (4/3) * density_value**(1/3)
        return 0.0
    
    # Початкова густина
    rho = [density(x, y, z) for x, y, z in grid_points]
    
    # Енергія Хартрі
    def hartree_energy(rho_values):
        energy = 0.0
        for i, (x1, y1, z1) in enumerate(grid_points):
            for j, (x2, y2, z2) in enumerate(grid_points):
                if i != j:
                    r = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
                    if r > 1e-6:  # уникнути ділення на нуль
                        energy += rho_values[i] * rho_values[j] / r
        return energy * 0.5  # множник 1/2
    
    # Повна енергія
    def total_energy(rho_values):
        # Кінетична енергія (приблизна)
        kinetic = sum(rho_val**2 for rho_val in rho_values)
        
        # Потенціальна енергія
        potential = sum(rho_val * external_potential(x, y, z) 
                       for rho_val, (x, y, z) in zip(rho_values, grid_points))
        
        # Енергія Хартрі
        hartree = hartree_energy(rho_values)
        
        # Обмінно-кореляційна енергія
        exchange_corr = sum(exchange_correlation_energy(rho_val) for rho_val in rho_values)
        
        return kinetic + potential + hartree + exchange_corr
    
    energies = []
    
    # Ітераційний процес Kohn-Sham
    for iteration in range(max_iterations):
        current_energy = total_energy(rho)
        energies.append(current_energy)
        
        # Оновлення густини (спрощений метод)
        # В реальних розрахунках потрібно розв'язувати рівняння Кона-Шам
        new_rho = []
        for rho_val in rho:
            # Просте оновлення
            updated_rho = rho_val * 0.9 + 0.1 * (rho_val + 0.01 * math.sin(iteration * 0.1))
            new_rho.append(max(0, updated_rho))  # густина не може бути від'ємною
        
        rho = new_rho
        
        # Перевірка збіжності
        if iteration > 0 and abs(energies[-1] - energies[-2]) < 1e-6:
            break
    
    return {
        "total_energy": energies[-1],
        "energies_history": energies,
        "final_density": rho,
        "converged": iteration < max_iterations - 1,
        "iterations": iteration + 1
    }

# Молекулярна динаміка
def molecular_dynamics_simulation(atom_positions: List[List[float]], 
                                atom_types: List[str], 
                                atom_masses: List[float], 
                                temperature: float, 
                                time_step: float, 
                                n_steps: int, 
                                thermostat: bool = True) -> Dict:
    """
    Молекулярна динаміка з інтегратором Верле.
    
    Параметри:
        atom_positions: Початкові позиції атомів
        atom_types: Типи атомів
        atom_masses: Маси атомів
        temperature: Температура (К)
        time_step: Крок часу (фс)
        n_steps: Кількість кроків
        thermostat: Використання термостата
    
    Повертає:
        Словник з результатами симуляції
    """
    # Константи
    k_B = constants.Boltzmann * 1e23  # Дж/K -> кДж/K
    fs_to_s = 1e-15  # фемтосекунди в секунди
    
    # Копія позицій
    positions = [row[:] for row in atom_positions]
    n_atoms = len(positions)
    
    # Початкові швидкості (Максвелл-Больцман)
    velocities = []
    for i in range(n_atoms):
        # Середньоквадратична швидкість
        mass = atom_masses[i] * 1e-3  # а.о.м. в кг
        rms_velocity = math.sqrt(3 * k_B * temperature / mass)
        
        # Випадкові компоненти швидкості
        vx = np.random.normal(0, rms_velocity / math.sqrt(3))
        vy = np.random.normal(0, rms_velocity / math.sqrt(3))
        vz = np.random.normal(0, rms_velocity / math.sqrt(3))
        velocities.append([vx, vy, vz])
    
    # Попередні позиції для інтегратора Верле
    prev_positions = []
    for i in range(n_atoms):
        prev_pos = []
        for j in range(3):
            # x(t-Δt) = x(t) - v(t) * Δt
            prev_pos.append(positions[i][j] - velocities[i][j] * time_step)
        prev_positions.append(prev_pos)
    
    # Функція сили (градієнт потенціальної енергії)
    def forces(pos):
        forces_list = []
        delta = 1e-4  # для чисельного диференціювання
        
        # Потенціальна енергія (спрощена модель)
        def potential(pos_config):
            energy = 0.0
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    # Відстань
                    r = math.sqrt(sum((pos_config[i][k] - pos_config[j][k])**2 for k in range(3)))
                    
                    # Потенціал Леннард-Джонса
                    sigma = 3.0
                    epsilon = 0.1
                    
                    if r > 0:
                        lj_energy = 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
                        energy += lj_energy
            
            return energy
        
        # Чисельне обчислення градієнта
        for i in range(n_atoms):
            force_atom = []
            for coord in range(3):
                # Зміщені конфігурації
                pos_plus = [row[:] for row in pos]
                pos_minus = [row[:] for row in pos]
                pos_plus[i][coord] += delta
                pos_minus[i][coord] -= delta
                
                # Градієнт (з протилежним знаком)
                grad = -(potential(pos_plus) - potential(pos_minus)) / (2 * delta)
                force_atom.append(grad)
            forces_list.append(force_atom)
        
        return forces_list
    
    # Збір статистики
    trajectory = [positions[:]]  # копія початкових позицій
    energies = []
    temperatures = []
    
    # Основний цикл молекулярної динаміки
    for step in range(n_steps):
        # Обчислення сил
        current_forces = forces(positions)
        
        # Інтегратор Верле
        new_positions = []
        for i in range(n_atoms):
            new_pos = []
            for j in range(3):
                # x(t+Δt) = 2*x(t) - x(t-Δt) + F(t)/m * (Δt)²
                mass = atom_masses[i] * 1e-3  # а.о.м. в кг
                acceleration = current_forces[i][j] / mass
                new_coord = 2 * positions[i][j] - prev_positions[i][j] + acceleration * (time_step**2)
                new_pos.append(new_coord)
            new_positions.append(new_pos)
        
        # Оновлення для наступного кроку
        prev_positions = [row[:] for row in positions]
        positions = new_positions
        
        # Термостат Бертрана (при необхідності)
        if thermostat and step > 0 and step % 10 == 0:
            # Обчислення поточної температури
            kinetic_energy = 0.0
            for i in range(n_atoms):
                mass = atom_masses[i] * 1e-3
                v_squared = sum(velocities[i][j]**2 for j in range(3))
                kinetic_energy += 0.5 * mass * v_squared
            
            current_temp = (2 * kinetic_energy) / (3 * n_atoms * k_B)
            
            # Масштабування швидкостей
            if current_temp > 0:
                scaling_factor = math.sqrt(temperature / current_temp)
                for i in range(n_atoms):
                    for j in range(3):
                        velocities[i][j] *= scaling_factor
        
        # Обчислення швидкостей для цього кроку
        for i in range(n_atoms):
            for j in range(3):
                # v(t) = [x(t+Δt) - x(t-Δt)] / (2*Δt)
                velocities[i][j] = (positions[i][j] - prev_positions[i][j]) / (2 * time_step)
        
        # Збір статистики кожні 10 кроків
        if step % 10 == 0:
            trajectory.append([row[:] for row in positions])
            
            # Потенціальна енергія
            potential_energy = 0.0
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    r = math.sqrt(sum((positions[i][k] - positions[j][k])**2 for k in range(3)))
                    sigma = 3.0
                    epsilon = 0.1
                    if r > 0:
                        potential_energy += 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
            
            # Кінетична енергія
            kinetic_energy = 0.0
            for i in range(n_atoms):
                mass = atom_masses[i] * 1e-3
                v_squared = sum(velocities[i][j]**2 for j in range(3))
                kinetic_energy += 0.5 * mass * v_squared
            
            total_energy = potential_energy + kinetic_energy
            energies.append(total_energy)
            
            # Температура
            temp = (2 * kinetic_energy) / (3 * n_atoms * k_B)
            temperatures.append(temp)
    
    return {
        "trajectory": trajectory,
        "energies": energies,
        "temperatures": temperatures,
        "final_positions": positions,
        "final_velocities": velocities
    }

# Спектроскопія
def uv_vis_spectrum(transition_energies: List[float], 
                   transition_intensities: List[float], 
                   broadening: float = 0.1) -> Callable[[float], float]:
    """
    Моделювання УФ-видимого спектра.
    
    Параметри:
        transition_energies: Енергії переходів (eV)
        transition_intensities: Інтенсивності переходів
        broadening: Параметр уширення ліній (eV)
    
    Повертає:
        Функція поглинання A(energy)
    """
    def absorption_spectrum(energy):
        absorption = 0.0
        for i, (e_trans, intensity) in enumerate(zip(transition_energies, transition_intensities)):
            # Гаусівське уширення
            gaussian = math.exp(-((energy - e_trans) / broadening)**2)
            absorption += intensity * gaussian
        return absorption
    
    return absorption_spectrum

def vibrational_spectrum(frequencies: List[float], 
                        intensities: List[float], 
                        temperature: float = 298.0) -> Callable[[float], float]:
    """
    Моделювання ІЧ-спектра коливань.
    
    Параметри:
        frequencies: Частоти коливань (см⁻¹)
        intensities: Інтенсивності коливань
        temperature: Температура (К)
    
    Повертає:
        Функція поглинання A(wavenumber)
    """
    k_B = constants.Boltzmann
    h = constants.Planck
    c = constants.c * 100  # м/с -> см/с
    
    def ir_spectrum(wavenumber):
        absorption = 0.0
        for i, (freq, intensity) in enumerate(zip(frequencies, intensities)):
            # Енергія коливання
            energy = h * c * freq
            
            # Популяція основного стану (фактор Больцмана)
            if temperature > 0:
                boltzmann_factor = math.exp(-energy / (k_B * temperature))
            else:
                boltzmann_factor = 1.0
            
            # Гаусівське уширення
            broadening = 10.0  # см⁻¹
            gaussian = math.exp(-((wavenumber - freq) / broadening)**2)
            
            absorption += intensity * boltzmann_factor * gaussian
        
        return absorption
    
    return ir_spectrum

# Реакційна здатність
def transition_state_theory(activation_energy: float, 
                          pre_exponential_factor: float, 
                          temperature: float) -> float:
    """
    Теорія переходу стану для обчислення константи швидкості.
    
    Параметри:
        activation_energy: Енергія активації (кДж/моль)
        pre_exponential_factor: Предекспоненціальний множник (с⁻¹)
        temperature: Температура (К)
    
    Повертає:
        Константа швидкості реакції (с⁻¹)
    """
    R = constants.R / 1000  # Дж/(моль·К) -> кДж/(моль·К)
    
    # Рівняння Арреніуса: k = A * exp(-Ea/RT)
    rate_constant = pre_exponential_factor * math.exp(-activation_energy / (R * temperature))
    
    return rate_constant

def reaction_coordinate_analysis(potential_energy_surface: Callable[[float], float], 
                               start_point: float, 
                               end_point: float, 
                               n_points: int = 1000) -> Dict:
    """
    Аналіз координати реакції.
    
    Параметри:
        potential_energy_surface: Потенційна енергія як функція координати реакції
        start_point: Початкова точка
        end_point: Кінцева точка
        n_points: Кількість точок
    
    Повертає:
        Словник з параметрами переходу
    """
    # Створення сітки точок
    coordinates = np.linspace(start_point, end_point, n_points)
    energies = [potential_energy_surface(coord) for coord in coordinates]
    
    # Знаходження максимуму (стан переходу)
    max_energy_index = np.argmax(energies)
    transition_state_energy = energies[max_energy_index]
    transition_state_coord = coordinates[max_energy_index]
    
    # Енергія активації
    reactant_energy = energies[0]
    product_energy = energies[-1]
    activation_energy = transition_state_energy - reactant_energy
    
    # Енергія реакції
    reaction_energy = product_energy - reactant_energy
    
    return {
        "activation_energy": activation_energy,
        "reaction_energy": reaction_energy,
        "transition_state_coordinate": transition_state_coord,
        "transition_state_energy": transition_state_energy,
        "coordinates": coordinates.tolist(),
        "energies": energies
    }

# Електрохімія
def nernst_equation(standard_potential: float, 
                   reactant_concentrations: List[float], 
                   product_concentrations: List[float], 
                   electron_transfer: int, 
                   temperature: float = 298.0) -> float:
    """
    Рівняння Нернста для обчислення електродного потенціалу.
    
    Параметри:
        standard_potential: Стандартний потенціал (В)
        reactant_concentrations: Концентрації реагентів
        product_concentrations: Концентрації продуктів
        electron_transfer: Кількість переданих електронів
        temperature: Температура (К)
    
    Повертає:
        Електродний потенціал (В)
    """
    R = constants.R
    F = constants.physical_constants['Faraday constant'][0]
    
    # Відношення концентрацій
    reactant_product = 1.0
    for conc in reactant_concentrations:
        reactant_product *= conc
    
    product_product = 1.0
    for conc in product_concentrations:
        product_product *= conc
    
    if reactant_product > 0:
        concentration_ratio = product_product / reactant_product
    else:
        concentration_ratio = float('inf')
    
    # Рівняння Нернста: E = E° - (RT/nF) * ln(Q)
    if concentration_ratio > 0:
        electrode_potential = standard_potential - (R * temperature / (electron_transfer * F)) * math.log(concentration_ratio)
    else:
        electrode_potential = standard_potential
    
    return electrode_potential

def cyclic_voltammetry(simulation_time: List[float], 
                      scan_rate: float, 
                      formal_potential: float, 
                      transfer_coefficient: float = 0.5) -> Tuple[List[float], List[float]]:
    """
    Симуляція циклічної вольтамперометрії.
    
    Параметри:
        simulation_time: Час симуляції
        scan_rate: Швидкість розгортки (В/с)
        formal_potential: Формальний потенціал (В)
        transfer_coefficient: Коефіцієнт переносу
    
    Повертає:
        Кортеж (potentials, currents) з потенціалами та струмами
    """
    R = constants.R
    F = constants.physical_constants['Faraday constant'][0]
    T = 298.0  # температура
    
    # Початковий потенціал
    initial_potential = formal_potential - 0.5  # В
    
    potentials = []
    currents = []
    
    for t in simulation_time:
        # Потенціал як функція часу (трикутна розгортка)
        if t < 1.0:
            # Пряме розгортання
            potential = initial_potential + scan_rate * t
        else:
            # Зворотне розгортання
            potential = initial_potential + scan_rate * 1.0 - scan_rate * (t - 1.0)
        
        potentials.append(potential)
        
        # Струм з рівняння Батлер-Вольмера
        # E = E° + (RT/αF) * ln((i₀ + i)/(i₀ - i))
        # Розв'язуємо відносно i
        
        # Приблизні параметри
        exchange_current = 1e-6  # A
        overpotential = potential - formal_potential
        
        # Спрощений вираз для струму
        if abs(overpotential) < 1e-10:
            current = 0.0
        else:
            # Лінійна апроксимація при малих overpotential
            current = exchange_current * overpotential * transfer_coefficient * F / (R * T)
        
        currents.append(current)
    
    return (potentials, currents)

# Хімічна термодинаміка
def chemical_equilibrium(initial_concentrations: List[float], 
                        equilibrium_constant: float, 
                        stoichiometric_coefficients: List[float]) -> List[float]:
    """
    Розрахунок рівноважних концентрацій.
    
    Параметри:
        initial_concentrations: Початкові концентрації
        equilibrium_constant: Константа рівноваги
        stoichiometric_coefficients: Стехіометричні коефіцієнти
    
    Повертає:
        Рівноважні концентрації
    """
    # Для реакції: aA + bB ⇌ cC + dD
    # K = [C]^c * [D]^d / ([A]^a * [B]^b)
    
    def equilibrium_expression(x):
        # x - зміна концентрації
        concentrations = []
        for i, initial in enumerate(initial_concentrations):
            conc = initial + stoichiometric_coefficients[i] * x
            concentrations.append(max(0, conc))  # концентрація не може бути від'ємною
        
        # Обчислення K з поточних концентрацій
        numerator = 1.0
        denominator = 1.0
        for i, conc in enumerate(concentrations):
            coeff = stoichiometric_coefficients[i]
            if coeff > 0:  # продукт
                numerator *= conc**coeff
            elif coeff < 0:  # реагент
                denominator *= conc**(-coeff)
        
        if denominator > 0:
            calculated_k = numerator / denominator
        else:
            calculated_k = float('inf')
        
        return abs(calculated_k - equilibrium_constant)
    
    # Знаходження рівноваги
    result = optimize.minimize_scalar(equilibrium_expression, bounds=(-1, 1), method='bounded')
    x_equilibrium = result.x
    
    # Рівноважні концентрації
    equilibrium_concentrations = []
    for i, initial in enumerate(initial_concentrations):
        conc = initial + stoichiometric_coefficients[i] * x_equilibrium
        equilibrium_concentrations.append(max(0, conc))
    
    return equilibrium_concentrations

def phase_diagram_analysis(phase_boundaries: List[Tuple[float, float]], 
                         temperature_range: Tuple[float, float], 
                         composition_range: Tuple[float, float]) -> Dict:
    """
    Аналіз діаграми фаз.
    
    Параметри:
        phase_boundaries: Межі фаз [(temp1, comp1), (temp2, comp2), ...]
        temperature_range: Діапазон температур
        composition_range: Діапазон складу
    
    Повертає:
        Словник з аналізом фазової діаграми
    """
    # Критичні точки
    critical_points = []
    
    # Аналіз меж фаз
    phase_regions = []
    
    # Евтектичні точки
    eutectic_points = []
    
    # Перитектичні точки
    peritectic_points = []
    
    # Спрощений аналіз
    temp_min, temp_max = temperature_range
    comp_min, comp_max = composition_range
    
    # Знаходження мінімальних та максимальних температур
    if phase_boundaries:
        min_temp = min(temp for temp, _ in phase_boundaries)
        max_temp = max(temp for temp, _ in phase_boundaries)
        min_comp = min(comp for _, comp in phase_boundaries)
        max_comp = max(comp for _, comp in phase_boundaries)
    else:
        min_temp, max_temp = temp_min, temp_max
        min_comp, max_comp = comp_min, comp_max
    
    return {
        "critical_points": critical_points,
        "phase_regions": phase_regions,
        "eutectic_points": eutectic_points,
        "peritectic_points": peritectic_points,
        "temperature_range": (min_temp, max_temp),
        "composition_range": (min_comp, max_comp)
    }

# Квантові хімічні розрахунки
def quantum_chemistry_integrals(basis_functions: List[Callable]) -> Dict:
    """
    Обчислення інтегралів в квантовій хімії.
    
    Параметри:
        basis_functions: Базисні функції
    
    Повертає:
        Словник з інтегралами
    """
    n_basis = len(basis_functions)
    
    # Інтеграли перекривання
    overlap_integrals = np.zeros((n_basis, n_basis))
    
    # Гамільтоніан одиничної частинки
    one_electron_integrals = np.zeros((n_basis, n_basis))
    
    # Двочастинкові інтеграли
    two_electron_integrals = np.zeros((n_basis, n_basis, n_basis, n_basis))
    
    # Спрощені обчислення (для демонстрації)
    for i in range(n_basis):
        for j in range(n_basis):
            # Приблизна оцінка інтегралів
            overlap_integrals[i, j] = 1.0 if i == j else 0.1
            one_electron_integrals[i, j] = -1.0 if i == j else 0.05
            
            for k in range(n_basis):
                for l in range(n_basis):
                    if i == j == k == l:
                        two_electron_integrals[i, j, k, l] = 1.0
                    elif (i == k and j == l) or (i == l and j == k):
                        two_electron_integrals[i, j, k, l] = 0.5
                    else:
                        two_electron_integrals[i, j, k, l] = 0.1
    
    return {
        "overlap_integrals": overlap_integrals.tolist(),
        "one_electron_integrals": one_electron_integrals.tolist(),
        "two_electron_integrals": two_electron_integrals.tolist()
    }

def configuration_interaction(coefficients: List[List[float]], 
                            orbital_energies: List[float], 
                            n_electrons: int) -> Dict:
    """
    Метод конфігураційної взаємодії (CI).
    
    Параметри:
        coefficients: Коефіцієнти молекулярних орбіталей
        orbital_energies: Енергії орбіталей
        n_electrons: Кількість електронів
    
    Повертає:
        Словник з результатами CI
    """
    # Кількість орбіталей
    n_orbitals = len(orbital_energies)
    n_occupied = n_electrons // 2  # заповнені орбіталі
    
    # Генерація конфігурацій (спрощена)
    configurations = []
    
    # Основна конфігурація
    ground_config = [1] * n_occupied + [0] * (n_orbitals - n_occupied)
    configurations.append(ground_config)
    
    # Збуджені конфігурації (одиночні збудження)
    for i in range(n_occupied):
        for a in range(n_occupied, n_orbitals):
            excited_config = ground_config[:]
            excited_config[i] = 0  # видалення електрона
            excited_config[a] = 1  # додавання електрона
            configurations.append(excited_config)
    
    # Матриця гамільтоніана CI
    n_configs = len(configurations)
    H_ci = np.zeros((n_configs, n_configs))
    
    # Заповнення матриці (спрощено)
    for i in range(n_configs):
        for j in range(n_configs):
            if i == j:
                # Діагональні елементи - енергія конфігурації
                energy = 0.0
                for orb in range(n_orbitals):
                    if configurations[i][orb] == 1:
                        energy += orbital_energies[orb]
                H_ci[i, j] = energy
            else:
                # Недіагональні елементи - взаємодія конфігурацій
                # Спрощена оцінка
                H_ci[i, j] = 0.1 if abs(i - j) == 1 else 0.0
    
    # Діагоналізація
    eigenvalues, eigenvectors = linalg.eigh(H_ci)
    
    return {
        "ci_energies": eigenvalues.tolist(),
        "ci_coefficients": eigenvectors.tolist(),
        "configurations": configurations,
        "n_configurations": n_configs
    }

# Статистична механіка молекул
def molecular_partition_function(translational_temperatures: List[float], 
                               rotational_temperatures: List[float], 
                               vibrational_temperatures: List[float], 
                               electronic_energies: List[float], 
                               temperature: float) -> float:
    """
    Обчислення функції розподілу молекули.
    
    Параметри:
        translational_temperatures: Температури перекладу
        rotational_temperatures: Температури обертання
        vibrational_temperatures: Температури коливань
        electronic_energies: Електронні енергії
        temperature: Температура (К)
    
    Повертає:
        Функція розподілу
    """
    # Перекладова частина
    q_trans = 1.0
    for theta in translational_temperatures:
        q_trans *= (2 * math.pi * constants.k * temperature / (constants.h**2))**1.5 * temperature / theta
    
    # Обертальна частина
    q_rot = 1.0
    for theta in rotational_temperatures:
        q_rot *= temperature / theta
    
    # Коливальна частина
    q_vib = 1.0
    for theta in vibrational_temperatures:
        if theta > 0:
            q_vib *= 1 / (1 - math.exp(-theta / temperature))
    
    # Електронна частина
    q_elec = 0.0
    for energy in electronic_energies:
        q_elec += math.exp(-energy / (constants.k * temperature))
    
    # Повна функція розподілу
    q_total = q_trans * q_rot * q_vib * q_elec
    
    return q_total

def thermodynamic_properties_from_partition(partition_function: Callable[[float], float], 
                                         temperature: float) -> Dict:
    """
    Термодинамічні властивості з функції розподілу.
    
    Параметри:
        partition_function: Функція розподілу Q(T)
        temperature: Температура (К)
    
    Повертає:
        Словник з термодинамічними властивостями
    """
    k_B = constants.Boltzmann
    
    # Чисельне диференціювання
    delta_t = 1e-3
    Q = partition_function(temperature)
    Q_plus = partition_function(temperature + delta_t)
    Q_minus = partition_function(temperature - delta_t)
    
    # Внутрішня енергія: U = kT² * d(lnQ)/dT
    lnQ = math.log(Q) if Q > 0 else 0
    lnQ_plus = math.log(Q_plus) if Q_plus > 0 else 0
    lnQ_minus = math.log(Q_minus) if Q_minus > 0 else 0
    dlnQ_dT = (lnQ_plus - lnQ_minus) / (2 * delta_t)
    internal_energy = k_B * temperature**2 * dlnQ_dT
    
    # Ентропія: S = k * (lnQ + T * d(lnQ)/dT)
    entropy = k_B * (lnQ + temperature * dlnQ_dT)
    
    # Вільна енергія Гельмгольца: F = -kT * lnQ
    helmholtz_free_energy = -k_B * temperature * lnQ
    
    # Теплоємність: Cv = k * T² * d²(lnQ)/dT²
    Q_plus2 = partition_function(temperature + 2*delta_t)
    Q_minus2 = partition_function(temperature - 2*delta_t)
    lnQ_plus2 = math.log(Q_plus2) if Q_plus2 > 0 else 0
    lnQ_minus2 = math.log(Q_minus2) if Q_minus2 > 0 else 0
    d2lnQ_dT2 = (lnQ_plus2 - 2*lnQ_plus + 2*lnQ_minus - lnQ_minus2) / (4 * delta_t**2)
    heat_capacity = k_B * temperature**2 * d2lnQ_dT2
    
    return {
        "internal_energy": internal_energy,
        "entropy": entropy,
        "helmholtz_free_energy": helmholtz_free_energy,
        "heat_capacity": heat_capacity,
        "partition_function": Q
    }

# Хімічна кінетика
def arrhenius_equation(pre_exponential_factor: float, 
                     activation_energy: float, 
                     temperature: float) -> float:
    """
    Рівняння Арреніуса для константи швидкості.
    
    Параметри:
        pre_exponential_factor: Предекспоненціальний множник
        activation_energy: Енергія активації (Дж/моль)
        temperature: Температура (К)
    
    Повертає:
        Константа швидкості
    """
    R = constants.R
    
    # k = A * exp(-Ea/RT)
    rate_constant = pre_exponential_factor * math.exp(-activation_energy / (R * temperature))
    
    return rate_constant

def reaction_mechanism_analysis(rate_constants: List[float], 
                               stoichiometry: List[List[float]]) -> Dict:
    """
    Аналіз механізму реакції.
    
    Параметри:
        rate_constants: Константи швидкостей елементарних реакцій
        stoichiometry: Стехіометрична матриця
    
    Повертає:
        Словник з аналізом механізму
    """
    # Кількість реакцій та речовин
    n_reactions = len(rate_constants)
    n_species = len(stoichiometry[0]) if stoichiometry else 0
    
    # Матриця стехіометрії
    nu = np.array(stoichiometry)
    
    # Швидкості реакцій
    reaction_rates = np.array(rate_constants)
    
    # Вектор швидкостей зміни концентрацій
    dC_dt = np.dot(nu.T, reaction_rates)
    
    # Знаходження швидких та повільних реакцій
    max_rate = max(rate_constants) if rate_constants else 1.0
    fast_reactions = [i for i, k in enumerate(rate_constants) if k > 0.1 * max_rate]
    slow_reactions = [i for i, k in enumerate(rate_constants) if k <= 0.1 * max_rate]
    
    # Визначення лімітуючого етапу
    rate_determining_step = rate_constants.index(min(rate_constants)) if rate_constants else 0
    
    return {
        "concentration_rates": dC_dt.tolist(),
        "fast_reactions": fast_reactions,
        "slow_reactions": slow_reactions,
        "rate_determining_step": rate_determining_step,
        "reaction_rates": rate_constants
    }

if __name__ == "__main__":
    # Тестування функцій модуля
    print("Тестування модуля обчислювальної хімії PyNexus")
    
    # Тест оптимізації геометрії
    atom_positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    atom_types = ["C", "H", "O"]
    optimized_positions, energy = molecular_geometry_optimizer(atom_positions, atom_types)
    print(f"Оптимізована енергія: {energy:.3f} ккал/моль")
    
    # Тест аналізу зв'язків
    bond_info = bond_analysis(optimized_positions, atom_types)
    print(f"Кількість зв'язків: {len(bond_info['bonds'])}")
    
    # Тест молекулярної динаміки
    atom_masses = [12.0, 1.0, 16.0]  # C, H, O
    md_results = molecular_dynamics_simulation(
        optimized_positions, atom_types, atom_masses, 
        temperature=300.0, time_step=1.0, n_steps=100
    )
    print(f"Кількість кроків МД: {len(md_results['energies'])}")
    
    # Тест рівняння Нернста
    potential = nernst_equation(0.34, [1.0], [1e-3], 2)
    print(f"Потенціал мідного електрода: {potential:.3f} В")
    
    # Тест рівняння Арреніуса
    k = arrhenius_equation(1e13, 50000, 300)
    print(f"Константа швидкості: {k:.3e} с⁻¹")
    
    print("Тестування завершено!")