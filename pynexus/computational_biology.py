"""
Модуль для обчислювальної біології в PyNexus.
Включає функції для моделювання біологічних систем, аналізу геноміки,
протеоміки, біоінформатики, еволюційної біології та системної біології.

Автор: Андрій Будильников
"""

import math
import numpy as np
from typing import List, Tuple, Callable, Union, Optional, Dict, Any
from scipy import constants, optimize, stats, signal
import matplotlib.pyplot as plt

# Геноміка та послідовності ДНК/РНК
def dna_sequence_analysis(dna_sequence: str) -> Dict[str, Any]:
    """
    Аналіз послідовності ДНК.
    
    Параметри:
        dna_sequence: Послідовність ДНК (A, T, G, C)
    
    Повертає:
        Словник з аналізом послідовності
    """
    # Перевірка валідності послідовності
    valid_bases = set('ATGC')
    if not all(base in valid_bases for base in dna_sequence.upper()):
        raise ValueError("Невалідна послідовність ДНК")
    
    sequence = dna_sequence.upper()
    length = len(sequence)
    
    # Підрахунок нуклеотидів
    nucleotide_counts = {
        'A': sequence.count('A'),
        'T': sequence.count('T'),
        'G': sequence.count('G'),
        'C': sequence.count('C')
    }
    
    # Частоти нуклеотидів
    nucleotide_frequencies = {base: count/length for base, count in nucleotide_counts.items()}
    
    # GC-вміст
    gc_content = (nucleotide_counts['G'] + nucleotide_counts['C']) / length
    
    # Комплементарна послідовність
    complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    complementary_sequence = ''.join(complement_map[base] for base in sequence)
    
    # Зворотна комплементарна послідовність
    reverse_complement = complementary_sequence[::-1]
    
    # Пошук ORF (відкритих рамок зчитування)
    def find_orfs(seq):
        orfs = []
        start_codon = 'ATG'
        stop_codons = ['TAA', 'TAG', 'TGA']
        
        for frame in range(3):
            i = frame
            while i < len(seq) - 2:
                if seq[i:i+3] == start_codon:
                    start = i
                    for j in range(i+3, len(seq) - 2, 3):
                        if seq[j:j+3] in stop_codons:
                            orfs.append((start, j+3, seq[start:j+3]))
                            i = j + 3
                            break
                    else:
                        i += 3
                else:
                    i += 3
        return orfs
    
    orfs = find_orfs(sequence)
    
    # Транскрипція (ДНК -> РНК)
    def transcribe(dna_seq):
        return dna_seq.replace('T', 'U')
    
    rna_sequence = transcribe(sequence)
    
    # Трансляція (РНК -> білки)
    genetic_code = {
        'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
        'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
        'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
        'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
        'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
        'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
        'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
        'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }
    
    def translate(rna_seq):
        protein = ""
        for i in range(0, len(rna_seq) - 2, 3):
            codon = rna_seq[i:i+3]
            if len(codon) == 3:
                amino_acid = genetic_code.get(codon, 'X')
                if amino_acid == '*':  # стоп-кодон
                    break
                protein += amino_acid
        return protein
    
    protein_sequence = translate(rna_sequence)
    
    return {
        'length': length,
        'nucleotide_counts': nucleotide_counts,
        'nucleotide_frequencies': nucleotide_frequencies,
        'gc_content': gc_content,
        'complementary_sequence': complementary_sequence,
        'reverse_complement': reverse_complement,
        'orfs': orfs,
        'rna_sequence': rna_sequence,
        'protein_sequence': protein_sequence
    }

def sequence_alignment(seq1: str, seq2: str, match_score: int = 2, 
                      mismatch_score: int = -1, gap_penalty: int = -1) -> Dict[str, Any]:
    """
    Вирівнювання двох послідовностей методом Нідлмана-Вунша.
    
    Параметри:
        seq1: Перша послідовність
        seq2: Друга послідовність
        match_score: Нагорода за збіг
        mismatch_score: Штраф за невідповідність
        gap_penalty: Штраф за пробіл
    
    Повертає:
        Словник з результатами вирівнювання
    """
    m, n = len(seq1), len(seq2)
    
    # Матриця оцінок
    score_matrix = np.zeros((m+1, n+1))
    
    # Ініціалізація
    for i in range(m+1):
        score_matrix[i][0] = gap_penalty * i
    for j in range(n+1):
        score_matrix[0][j] = gap_penalty * j
    
    # Заповнення матриці
    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq1[i-1] == seq2[j-1]:
                diagonal = score_matrix[i-1][j-1] + match_score
            else:
                diagonal = score_matrix[i-1][j-1] + mismatch_score
            
            up = score_matrix[i-1][j] + gap_penalty
            left = score_matrix[i][j-1] + gap_penalty
            
            score_matrix[i][j] = max(diagonal, up, left)
    
    # Відтворення вирівнювання
    align1, align2 = "", ""
    i, j = m, n
    
    while i > 0 or j > 0:
        current_score = score_matrix[i][j]
        
        if i > 0 and j > 0:
            if seq1[i-1] == seq2[j-1]:
                diagonal_score = score_matrix[i-1][j-1] + match_score
            else:
                diagonal_score = score_matrix[i-1][j-1] + mismatch_score
            
            if current_score == diagonal_score:
                align1 = seq1[i-1] + align1
                align2 = seq2[j-1] + align2
                i -= 1
                j -= 1
                continue
        
        if i > 0 and current_score == score_matrix[i-1][j] + gap_penalty:
            align1 = seq1[i-1] + align1
            align2 = "-" + align2
            i -= 1
        elif j > 0 and current_score == score_matrix[i][j-1] + gap_penalty:
            align1 = "-" + align1
            align2 = seq2[j-1] + align2
            j -= 1
    
    # Обчислення статистик
    matches = sum(1 for a, b in zip(align1, align2) if a == b and a != '-')
    mismatches = sum(1 for a, b in zip(align1, align2) if a != b and a != '-' and b != '-')
    gaps = align1.count('-') + align2.count('-')
    
    alignment_score = score_matrix[m][n]
    
    return {
        'alignment_score': alignment_score,
        'aligned_seq1': align1,
        'aligned_seq2': align2,
        'matches': matches,
        'mismatches': mismatches,
        'gaps': gaps,
        'identity': matches / len(align1) if len(align1) > 0 else 0,
        'similarity': (matches + mismatches) / len(align1) if len(align1) > 0 else 0
    }

def phylogenetic_tree_analysis(distance_matrix: List[List[float]], 
                             species_names: List[str]) -> Dict[str, Any]:
    """
    Простий аналіз філогенетичного дерева методом UPGMA.
    
    Параметри:
        distance_matrix: Матриця відстаней між видами
        species_names: Назви видів
    
    Повертає:
        Словник з філогенетичним деревом
    """
    n = len(species_names)
    
    # Створення копії матриці відстаней
    distances = [row[:] for row in distance_matrix]
    
    # Список кластерів
    clusters = [[name] for name in species_names]
    
    # Історія об'єднань
    merge_history = []
    
    # UPGMA алгоритм
    while len(clusters) > 1:
        # Знаходження мінімальної відстані
        min_distance = float('inf')
        merge_i, merge_j = 0, 1
        
        for i in range(len(distances)):
            for j in range(i+1, len(distances)):
                if distances[i][j] < min_distance:
                    min_distance = distances[i][j]
                    merge_i, merge_j = i, j
        
        # Об'єднання кластерів
        new_cluster = clusters[merge_i] + clusters[merge_j]
        merge_history.append({
            'clusters': (clusters[merge_i][:], clusters[merge_j][:]),
            'distance': min_distance,
            'new_cluster': new_cluster[:]
        })
        
        # Створення нової матриці відстаней
        new_distances = []
        new_clusters = []
        
        # Додавання нового кластера
        new_clusters.append(new_cluster)
        
        # Додавання інших кластерів
        for i in range(len(clusters)):
            if i != merge_i and i != merge_j:
                new_clusters.append(clusters[i])
        
        # Обчислення нових відстаней
        for i in range(len(new_clusters)):
            new_row = []
            for j in range(len(new_clusters)):
                if i == j:
                    new_row.append(0.0)
                elif i == 0:  # новий кластер
                    # Середня відстань до кластера j
                    total_distance = 0.0
                    count = 0
                    for species1 in new_cluster:
                        for species2 in new_clusters[j]:
                            idx1 = species_names.index(species1)
                            idx2 = species_names.index(species2)
                            total_distance += distance_matrix[idx1][idx2]
                            count += 1
                    avg_distance = total_distance / count if count > 0 else 0.0
                    new_row.append(avg_distance)
                elif j == 0:  # симетрія
                    new_row.append(new_distances[0][i])
                else:
                    # Відстань між існуючими кластерами
                    idx1 = [k for k, cl in enumerate(clusters) if cl == new_clusters[i]][0]
                    idx2 = [k for k, cl in enumerate(clusters) if cl == new_clusters[j]][0]
                    new_row.append(distances[idx1][idx2])
            new_distances.append(new_row)
        
        clusters = new_clusters
        distances = new_distances
    
    return {
        'tree': merge_history,
        'root_cluster': clusters[0] if clusters else [],
        'n_merges': len(merge_history)
    }

# Протеоміка та структурна біологія
def protein_structure_analysis(amino_acid_sequence: str) -> Dict[str, Any]:
    """
    Аналіз властивостей білка на основі амінокислотної послідовності.
    
    Параметри:
        amino_acid_sequence: Послідовність амінокислот
    
    Повертає:
        Словник з властивостями білка
    """
    # Властивості амінокислот
    amino_acid_properties = {
        'A': {'hydrophobic': True, 'polar': False, 'charged': False, 'weight': 89.09},
        'R': {'hydrophobic': False, 'polar': True, 'charged': True, 'weight': 174.20},
        'N': {'hydrophobic': False, 'polar': True, 'charged': False, 'weight': 132.12},
        'D': {'hydrophobic': False, 'polar': True, 'charged': True, 'weight': 133.10},
        'C': {'hydrophobic': True, 'polar': True, 'charged': False, 'weight': 121.16},
        'E': {'hydrophobic': False, 'polar': True, 'charged': True, 'weight': 147.13},
        'Q': {'hydrophobic': False, 'polar': True, 'charged': False, 'weight': 146.15},
        'G': {'hydrophobic': False, 'polar': False, 'charged': False, 'weight': 75.07},
        'H': {'hydrophobic': False, 'polar': True, 'charged': True, 'weight': 155.16},
        'I': {'hydrophobic': True, 'polar': False, 'charged': False, 'weight': 131.17},
        'L': {'hydrophobic': True, 'polar': False, 'charged': False, 'weight': 131.17},
        'K': {'hydrophobic': False, 'polar': True, 'charged': True, 'weight': 146.19},
        'M': {'hydrophobic': True, 'polar': False, 'charged': False, 'weight': 149.21},
        'F': {'hydrophobic': True, 'polar': False, 'charged': False, 'weight': 165.19},
        'P': {'hydrophobic': False, 'polar': False, 'charged': False, 'weight': 115.13},
        'S': {'hydrophobic': False, 'polar': True, 'charged': False, 'weight': 105.09},
        'T': {'hydrophobic': False, 'polar': True, 'charged': False, 'weight': 119.12},
        'W': {'hydrophobic': True, 'polar': False, 'charged': False, 'weight': 204.23},
        'Y': {'hydrophobic': True, 'polar': True, 'charged': False, 'weight': 181.19},
        'V': {'hydrophobic': True, 'polar': False, 'charged': False, 'weight': 117.15}
    }
    
    sequence = amino_acid_sequence.upper()
    length = len(sequence)
    
    # Підрахунок амінокислот
    aa_counts = {}
    for aa in sequence:
        if aa in amino_acid_properties:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
    
    # Молекулярна маса
    molecular_weight = sum(amino_acid_properties.get(aa, {'weight': 0})['weight'] for aa in sequence)
    # Віднімаємо масу води за кожний пептидний зв'язок
    molecular_weight -= (length - 1) * 18.015
    
    # Гідрофобність
    hydrophobic_count = sum(1 for aa in sequence if amino_acid_properties.get(aa, {}).get('hydrophobic', False))
    hydrophobicity = hydrophobic_count / length if length > 0 else 0
    
    # Полярність
    polar_count = sum(1 for aa in sequence if amino_acid_properties.get(aa, {}).get('polar', False))
    polarity = polar_count / length if length > 0 else 0
    
    # Зарядженість
    charged_count = sum(1 for aa in sequence if amino_acid_properties.get(aa, {}).get('charged', False))
    charge = charged_count / length if length > 0 else 0
    
    # Ізoeлектрична точка (спрощений розрахунок)
    # Позитивно заряджені: R, K, H
    # Негативно заряджені: D, E
    positive_count = sequence.count('R') + sequence.count('K') + sequence.count('H')
    negative_count = sequence.count('D') + sequence.count('E')
    isoelectric_point = 7.0 + (positive_count - negative_count) * 0.1
    
    # Прогноз вторинної структури (спрощений)
    def predict_secondary_structure(seq):
        alpha_helix = 0
        beta_sheet = 0
        coil = 0
        
        # Амінокислоти, що схильні до α-спіралі
        helix_favoring = set('ALIFEK')
        # Амінокислоти, що схильні до β-структури
        sheet_favoring = set('VITMC')
        
        for aa in seq:
            if aa in helix_favoring:
                alpha_helix += 1
            elif aa in sheet_favoring:
                beta_sheet += 1
            else:
                coil += 1
        
        total = len(seq)
        return {
            'alpha_helix': alpha_helix / total if total > 0 else 0,
            'beta_sheet': beta_sheet / total if total > 0 else 0,
            'coil': coil / total if total > 0 else 0
        }
    
    secondary_structure = predict_secondary_structure(sequence)
    
    return {
        'length': length,
        'amino_acid_counts': aa_counts,
        'molecular_weight': molecular_weight,
        'hydrophobicity': hydrophobicity,
        'polarity': polarity,
        'charge': charge,
        'isoelectric_point': isoelectric_point,
        'secondary_structure': secondary_structure
    }

def protein_folding_simulation(amino_acid_sequence: str, 
                             n_steps: int = 1000) -> Dict[str, Any]:
    """
    Проста симуляція згортання білка методом Монте-Карло.
    
    Параметри:
        amino_acid_sequence: Послідовність амінокислот
        n_steps: Кількість кроків симуляції
    
    Повертає:
        Словник з результатами симуляції
    """
    # Гідрофобні амінокислоти
    hydrophobic = set('AILFVPGMW')
    
    # Початкова конформація (лінійна)
    n = len(amino_acid_sequence)
    positions = [(i, 0) for i in range(n)]  # 2D сітка
    
    # Функція енергії
    def energy(pos):
        total_energy = 0
        
        # Взаємодії між амінокислотами
        for i in range(n):
            for j in range(i+2, n):  # не сусідні амінокислоти
                # Відстань між амінокислотами
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Якщо амінокислоти поруч (відстань = 1)
                if abs(distance - 1.0) < 1e-6:
                    # Гідрофобні взаємодії
                    if amino_acid_sequence[i] in hydrophobic and amino_acid_sequence[j] in hydrophobic:
                        total_energy -= 1.0  # приваблювання
                    elif (amino_acid_sequence[i] in hydrophobic) != (amino_acid_sequence[j] in hydrophobic):
                        total_energy += 0.5  # відштовхування
        
        return total_energy
    
    # Перевірка валідності конформації
    def is_valid_configuration(pos):
        # Перевірка самоперетинів
        for i in range(n):
            for j in range(i+1, n):
                if pos[i] == pos[j]:
                    return False
        return True
    
    # Початкова енергія
    current_energy = energy(positions)
    energies = [current_energy]
    best_energy = current_energy
    best_positions = positions[:]
    
    # Монте-Карло симуляція
    for step in range(n_steps):
        # Випадковий вибір амінокислоти для переміщення
        move_index = np.random.randint(1, n-1)  # не переміщуємо кінці
        
        # Випадковий вибір нового положення
        old_pos = positions[move_index]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dx, dy = directions[np.random.randint(4)]
        new_pos = (old_pos[0] + dx, old_pos[1] + dy)
        
        # Перевірка, чи нова позиція вільна
        if new_pos not in positions:
            # Тимчасове переміщення
            positions[move_index] = new_pos
            
            # Перевірка валідності
            if is_valid_configuration(positions):
                # Обчислення нової енергії
                new_energy = energy(positions)
                
                # Прийняття переміщення згідно з алгоритмом Метрополіса
                delta_energy = new_energy - current_energy
                temperature = 1.0 - (step / n_steps) * 0.9  # охолодження
                
                if delta_energy < 0 or np.random.random() < math.exp(-delta_energy / temperature):
                    # Прийняти переміщення
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_positions = positions[:]
                else:
                    # Повернути попередню позицію
                    positions[move_index] = old_pos
            else:
                # Повернути попередню позицію
                positions[move_index] = old_pos
        
        energies.append(current_energy)
    
    # Аналіз фінальної структури
    # Радіус гіратії
    center_of_mass = [sum(p[0] for p in best_positions) / n, 
                      sum(p[1] for p in best_positions) / n]
    radius_of_gyration = math.sqrt(sum((p[0] - center_of_mass[0])**2 + (p[1] - center_of_mass[1])**2 
                                      for p in best_positions) / n)
    
    # Кількість контаків
    contacts = 0
    for i in range(n):
        for j in range(i+2, n):
            dx = best_positions[i][0] - best_positions[j][0]
            dy = best_positions[i][1] - best_positions[j][1]
            distance = math.sqrt(dx*dx + dy*dy)
            if abs(distance - 1.0) < 1e-6:
                contacts += 1
    
    return {
        'final_energy': best_energy,
        'initial_energy': energies[0],
        'energy_history': energies,
        'final_positions': best_positions,
        'radius_of_gyration': radius_of_gyration,
        'contacts': contacts,
        'folding_progress': (energies[0] - best_energy) / abs(energies[0]) if energies[0] != 0 else 0
    }

# Системна біологія та мережі
def gene_regulatory_network(gene_expression: List[float], 
                          interaction_matrix: List[List[float]], 
                          time_points: List[float]) -> Dict[str, Any]:
    """
    Моделювання генно-регуляторної мережі.
    
    Параметри:
        gene_expression: Початкові рівні експресії генів
        interaction_matrix: Матриця взаємодій між генами
        time_points: Точки часу для симуляції
    
    Повертає:
        Словник з результатами симуляції
    """
    n_genes = len(gene_expression)
    
    # Модель диференціальних рівнянь
    def gene_dynamics(expression_levels, t):
        derivatives = []
        for i in range(n_genes):
            # Базовий рівень експресії
            basal = 0.1
            
            # Вплив інших генів
            interaction = 0.0
            for j in range(n_genes):
                interaction += interaction_matrix[i][j] * expression_levels[j]
            
            # Нелінійна функція відгуку (гіперболічний тангенс)
            regulation = math.tanh(interaction)
            
            # Деградація
            degradation = -0.05 * expression_levels[i]
            
            derivative = basal + regulation + degradation
            derivatives.append(derivative)
        
        return derivatives
    
    # Чисельне інтегрування (метод Ейлера)
    dt = time_points[1] - time_points[0] if len(time_points) > 1 else 0.1
    expression_history = [gene_expression[:]]
    current_expression = gene_expression[:]
    
    for t in time_points[1:]:
        derivatives = gene_dynamics(current_expression, t)
        new_expression = [current_expression[i] + dt * derivatives[i] for i in range(n_genes)]
        # Обмеження рівнів експресії
        new_expression = [max(0, min(10, expr)) for expr in new_expression]
        current_expression = new_expression
        expression_history.append(current_expression[:])
    
    # Аналіз мережі
    # Ступені вузлів
    in_degrees = [sum(1 for row in interaction_matrix if row[i] != 0) for i in range(n_genes)]
    out_degrees = [sum(1 for val in row if val != 0) for row in interaction_matrix]
    
    # Аттрактори (стійкі стани)
    def find_attractors(history):
        attractors = []
        threshold = 1e-3
        
        for i in range(1, len(history)):
            # Перевірка, чи система стабілізувалась
            diff = sum(abs(history[i][j] - history[i-1][j]) for j in range(n_genes))
            if diff < threshold:
                attractors.append(history[i][:])
                break
        
        return attractors
    
    attractors = find_attractors(expression_history)
    
    return {
        'expression_dynamics': expression_history,
        'attractors': attractors,
        'in_degrees': in_degrees,
        'out_degrees': out_degrees,
        'network_size': n_genes,
        'simulation_time': len(time_points)
    }

def metabolic_network_analysis(stoichiometry_matrix: List[List[float]], 
                             reaction_rates: List[float]) -> Dict[str, Any]:
    """
    Аналіз метаболічної мережі.
    
    Параметри:
        stoichiometry_matrix: Стехіометрична матриця (метаболіти x реакції)
        reaction_rates: Швидкості реакцій
    
    Повертає:
        Словник з аналізом метаболічної мережі
    """
    # Перетворення в numpy масиви
    S = np.array(stoichiometry_matrix)
    v = np.array(reaction_rates)
    
    # Вектор швидкостей зміни концентрацій: dC/dt = S * v
    dC_dt = np.dot(S, v)
    
    # Знаходження нульового простору (стійкі стани)
    # S * v = 0
    try:
        # Знаходження базису нульового простору
        null_space = np.linalg.null_space(S) if hasattr(np.linalg, 'null_space') else \
                     np.linalg.svd(S)[2].T[:, -1]  # спрощений підхід
    except:
        null_space = np.zeros(len(reaction_rates))
    
    # Аналіз зв'язності
    n_metabolites, n_reactions = S.shape
    
    # Ступені вузлів
    metabolite_degrees = [np.count_nonzero(S[i, :]) for i in range(n_metabolites)]
    reaction_degrees = [np.count_nonzero(S[:, j]) for j in range(n_reactions)]
    
    # Коефіцієнт участі метаболітів
    participation_coefficients = []
    for i in range(n_metabolites):
        total_flux = sum(abs(S[i, j] * reaction_rates[j]) for j in range(n_reactions))
        participation_coefficients.append(total_flux)
    
    # Ефективність мережі
    total_activity = sum(abs(rate) for rate in reaction_rates)
    network_efficiency = sum(dC_dt**2)**0.5 / total_activity if total_activity > 0 else 0
    
    return {
        'concentration_rates': dC_dt.tolist(),
        'null_space': null_space.tolist(),
        'metabolite_degrees': metabolite_degrees,
        'reaction_degrees': reaction_degrees,
        'participation_coefficients': participation_coefficients,
        'network_efficiency': network_efficiency,
        'n_metabolites': n_metabolites,
        'n_reactions': n_reactions
    }

# Еволюційна біологія
def population_genetics_simulation(initial_allele_frequencies: List[float], 
                                 fitness_values: List[float], 
                                 population_size: int, 
                                 generations: int) -> Dict[str, Any]:
    """
    Симуляція популяційної генетики.
    
    Параметри:
        initial_allele_frequencies: Початкові частоти алелів
        fitness_values: Значення пристосованості для кожного генотипу
        population_size: Розмір популяції
        generations: Кількість поколінь
    
    Повертає:
        Словник з результатами симуляції
    """
    # Для двоалельної системи (A і a)
    p = initial_allele_frequencies[0]  # частота алеля A
    q = 1 - p  # частота алеля a
    
    # Генотипи: AA, Aa, aa
    genotype_frequencies = [p*p, 2*p*q, q*q]  # згідно з рівнянням Харді-Вайнберга
    
    frequency_history = [[p, q]]
    genotype_history = [genotype_frequencies[:]]
    
    for generation in range(generations):
        # Очікувана пристосованість
        mean_fitness = sum(genotype_frequencies[i] * fitness_values[i] for i in range(3))
        
        if mean_fitness == 0:
            break
        
        # Нові частоти алелів після відбору
        new_AA_freq = genotype_frequencies[0] * fitness_values[0] / mean_fitness
        new_Aa_freq = genotype_frequencies[1] * fitness_values[1] / mean_fitness
        new_aa_freq = genotype_frequencies[2] * fitness_values[2] / mean_fitness
        
        # Нові частоти алелів
        new_p = new_AA_freq + 0.5 * new_Aa_freq
        new_q = new_aa_freq + 0.5 * new_Aa_freq
        
        # Генетичний дрейф (ефект конечной популяції)
        if population_size < 1000:  # тільки для малих популяцій
            drift_effect = np.random.normal(0, 0.01)
            new_p = max(0, min(1, new_p + drift_effect))
            new_q = 1 - new_p
        
        p, q = new_p, new_q
        genotype_frequencies = [p*p, 2*p*q, q*q]
        
        frequency_history.append([p, q])
        genotype_history.append(genotype_frequencies[:])
    
    # Рівноважні частоти
    equilibrium_p = frequency_history[-1][0] if frequency_history else p
    equilibrium_q = frequency_history[-1][1] if frequency_history else q
    
    # Час до рівноваги
    equilibrium_generation = len(frequency_history) - 1
    
    return {
        'allele_frequencies': frequency_history,
        'genotype_frequencies': genotype_history,
        'equilibrium_p': equilibrium_p,
        'equilibrium_q': equilibrium_q,
        'equilibrium_generation': equilibrium_generation,
        'fixation_probability': 1.0 if equilibrium_p > 0.99 else (0.0 if equilibrium_p < 0.01 else None)
    }

def molecular_evolution_analysis(sequences: List[str], 
                               time_points: List[float]) -> Dict[str, Any]:
    """
    Аналіз молекулярної еволюції.
    
    Параметри:
        sequences: Послідовності ДНК/білків у різні моменти часу
        time_points: Точки часу
    
    Повертає:
        Словник з аналізом молекулярної еволюції
    """
    if len(sequences) < 2:
        raise ValueError("Потрібно щонайменше дві послідовності")
    
    # Обчислення відстаней між послідовностями
    def sequence_distance(seq1, seq2):
        if len(seq1) != len(seq2):
            # Вирівнювання послідовностей
            alignment = sequence_alignment(seq1, seq2)
            seq1_aligned = alignment['aligned_seq1']
            seq2_aligned = alignment['aligned_seq2']
        else:
            seq1_aligned, seq2_aligned = seq1, seq2
        
        # Кількість замін
        differences = sum(1 for a, b in zip(seq1_aligned, seq2_aligned) if a != b and a != '-' and b != '-')
        total_positions = len(seq1_aligned) - seq1_aligned.count('-') - seq2_aligned.count('-')
        
        return differences / total_positions if total_positions > 0 else 0
    
    # Матриця відстаней
    n_sequences = len(sequences)
    distance_matrix = []
    
    for i in range(n_sequences):
        row = []
        for j in range(n_sequences):
            if i == j:
                row.append(0.0)
            else:
                dist = sequence_distance(sequences[i], sequences[j])
                row.append(dist)
        distance_matrix.append(row)
    
    # Швидкість молекулярної еволюції
    if len(time_points) >= 2:
        # Лінійна регресія для оцінки швидкості
        distances_from_first = [distance_matrix[0][i] for i in range(n_sequences)]
        
        # Метод найменших квадратів
        n = len(time_points)
        sum_x = sum(time_points)
        sum_y = sum(distances_from_first)
        sum_xy = sum(time_points[i] * distances_from_first[i] for i in range(n))
        sum_x2 = sum(t**2 for t in time_points)
        
        if n * sum_x2 - sum_x**2 != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            intercept = (sum_y - slope * sum_x) / n
        else:
            slope = 0
            intercept = 0
        
        evolutionary_rate = slope
    else:
        evolutionary_rate = 0
    
    # Аналіз позитивного відбору (dN/dS)
    # Спрощений підхід
    synonymous_substitutions = 0.1  # приблизне значення
    non_synonymous_substitutions = evolutionary_rate * 2  # приблизне значення
    
    if synonymous_substitutions > 0:
        dn_ds_ratio = non_synonymous_substitutions / synonymous_substitutions
    else:
        dn_ds_ratio = float('inf')
    
    # Інтерпретація dN/dS
    if dn_ds_ratio < 1:
        selection_type = "purifying"  # очищуючий відбір
    elif dn_ds_ratio > 1:
        selection_type = "positive"   # позитивний відбір
    else:
        selection_type = "neutral"    # нейтральна еволюція
    
    return {
        'distance_matrix': distance_matrix,
        'evolutionary_rate': evolutionary_rate,
        'dn_ds_ratio': dn_ds_ratio,
        'selection_type': selection_type,
        'sequence_length': len(sequences[0]) if sequences else 0,
        'n_sequences': n_sequences
    }

# Біоінформатика та обробка даних
def microarray_data_analysis(expression_data: List[List[float]], 
                           sample_labels: List[str]) -> Dict[str, Any]:
    """
    Аналіз даних мікроarray.
    
    Параметри:
        expression_data: Матриця експресії генів (гени x зразки)
        sample_labels: Мітки зразків
    
    Повертає:
        Словник з аналізом експресії
    """
    # Перетворення в numpy масив
    data = np.array(expression_data)
    n_genes, n_samples = data.shape
    
    # Базова статистика
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)
    medians = np.median(data, axis=1)
    
    # Нормалізація даних (Z-оцінка)
    normalized_data = (data - means[:, np.newaxis]) / (stds[:, np.newaxis] + 1e-10)
    
    # Диференціальна експресія (t-тест між групами)
    # Припустимо, перші n_samples//2 зразків - контрольна група
    control_group = data[:, :n_samples//2]
    experimental_group = data[:, n_samples//2:]
    
    # t-тест для кожного гена
    p_values = []
    fold_changes = []
    
    for i in range(n_genes):
        control_mean = np.mean(control_group[i, :])
        exp_mean = np.mean(experimental_group[i, :])
        
        # Fold change
        if control_mean > 0:
            fold_change = exp_mean / control_mean
        else:
            fold_change = float('inf') if exp_mean > 0 else 1.0
        fold_changes.append(fold_change)
        
        # t-тест
        try:
            t_stat, p_val = stats.ttest_ind(control_group[i, :], experimental_group[i, :])
            p_values.append(p_val)
        except:
            p_values.append(1.0)
    
    # Корекція p-значень (метод Бенжаміні-Хохберга)
    def benjamini_hochberg_correction(p_vals):
        n = len(p_vals)
        sorted_p_vals = sorted(p_vals)
        corrected_p_vals = [p * n / (i + 1) for i, p in enumerate(sorted_p_vals)]
        # Обмеження значень
        for i in range(n-2, -1, -1):
            corrected_p_vals[i] = min(corrected_p_vals[i], corrected_p_vals[i+1])
        return corrected_p_vals
    
    corrected_p_values = benjamini_hochberg_correction(p_values)
    
    # Значимо диференційовані гени (p < 0.05, fold change > 2 або < 0.5)
    significant_genes = []
    for i in range(n_genes):
        if corrected_p_values[i] < 0.05 and (fold_changes[i] > 2 or fold_changes[i] < 0.5):
            significant_genes.append({
                'gene_index': i,
                'fold_change': fold_changes[i],
                'p_value': corrected_p_values[i],
                'mean_expression': means[i]
            })
    
    # Кластеризація зразків
    # Евклідова відстань між зразками
    sample_distances = []
    for i in range(n_samples):
        row = []
        for j in range(n_samples):
            if i == j:
                row.append(0.0)
            else:
                dist = np.linalg.norm(data[:, i] - data[:, j])
                row.append(dist)
        sample_distances.append(row)
    
    # PCA аналіз (спрощений)
    # Центрування даних
    centered_data = data - np.mean(data, axis=1)[:, np.newaxis]
    
    # Сингулярне розкладання
    try:
        U, s, Vt = np.linalg.svd(centered_data, full_matrices=False)
        # Перші дві головні компоненти
        pc1 = Vt[0, :]
        pc2 = Vt[1, :]
    except:
        pc1 = np.zeros(n_samples)
        pc2 = np.zeros(n_samples)
    
    return {
        'means': means.tolist(),
        'stds': stds.tolist(),
        'medians': medians.tolist(),
        'normalized_data': normalized_data.tolist(),
        'fold_changes': fold_changes,
        'p_values': corrected_p_values,
        'significant_genes': significant_genes,
        'sample_distances': sample_distances,
        'principal_components': [pc1.tolist(), pc2.tolist()],
        'n_genes': n_genes,
        'n_samples': n_samples
    }

def protein_structure_prediction(contact_map: List[List[float]], 
                               sequence: str, 
                               n_iterations: int = 1000) -> Dict[str, Any]:
    """
    Прогноз структури білка на основі карти контактів.
    
    Параметри:
        contact_map: Матриця контактів між амінокислотами
        sequence: Амінокислотна послідовність
        n_iterations: Кількість ітерацій оптимізації
    
    Повертає:
        Словник з прогнозованою структурою
    """
    n = len(sequence)
    contacts = np.array(contact_map)
    
    # Початкова 3D структура (спіраль)
    phi = np.linspace(0, 4*np.pi, n)
    x = np.cos(phi)
    y = np.sin(phi)
    z = np.linspace(0, 10, n)
    
    positions = np.array([x, y, z]).T
    
    # Функція енергії
    def energy(pos):
        total_energy = 0
        
        for i in range(n):
            for j in range(i+1, n):
                # Відстань між атомами
                dist = np.linalg.norm(pos[i] - pos[j])
                
                # Ідеальна відстань для контакту
                ideal_dist = 3.8  # Ангстрем
                
                # Енергія взаємодії
                if contacts[i, j] > 0.5:  # контакт
                    # Гармонічний потенціал
                    total_energy += (dist - ideal_dist)**2
                else:
                    # Відштовхування при близькій відстані
                    if dist < 3.0:
                        total_energy += 10.0 / (dist + 0.1)
        
        return total_energy
    
    # Градієнт енергії
    def gradient(pos):
        grad = np.zeros_like(pos)
        delta = 1e-6
        
        for i in range(n):
            for j in range(3):  # x, y, z
                pos_plus = pos.copy()
                pos_minus = pos.copy()
                pos_plus[i, j] += delta
                pos_minus[i, j] -= delta
                
                grad[i, j] = (energy(pos_plus) - energy(pos_minus)) / (2 * delta)
        
        return grad
    
    # Оптимізація структури (градієнтний спуск)
    learning_rate = 0.01
    energies = [energy(positions)]
    
    for iteration in range(n_iterations):
        grad = gradient(positions)
        positions -= learning_rate * grad
        
        # Обмеження руху
        if iteration % 100 == 0:
            current_energy = energy(positions)
            energies.append(current_energy)
            
            # Адаптація швидкості навчання
            if len(energies) > 1 and energies[-1] > energies[-2]:
                learning_rate *= 0.9
    
    # Аналіз фінальної структури
    # Радіус гіратії
    center_of_mass = np.mean(positions, axis=0)
    radius_of_gyration = np.sqrt(np.mean(np.sum((positions - center_of_mass)**2, axis=1)))
    
    # Кількість реальних контактів
    actual_contacts = 0
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if contacts[i, j] > 0.5 and dist < 5.0:  # 5 Ангстрем
                actual_contacts += 1
    
    # RMSD від початкової структури
    initial_positions = np.array([np.cos(np.linspace(0, 4*np.pi, n)), 
                                 np.sin(np.linspace(0, 4*np.pi, n)), 
                                 np.linspace(0, 10, n)]).T
    rmsd = np.sqrt(np.mean(np.sum((positions - initial_positions)**2, axis=1)))
    
    return {
        'final_positions': positions.tolist(),
        'energy_history': energies,
        'radius_of_gyration': float(radius_of_gyration),
        'actual_contacts': actual_contacts,
        'rmsd': float(rmsd),
        'sequence_length': n
    }

# Нейробіологія
def neural_network_simulation(neuron_connections: List[List[float]], 
                            input_signals: List[float], 
                            simulation_time: List[float]) -> Dict[str, Any]:
    """
    Симуляція нейронної мережі.
    
    Параметри:
        neuron_connections: Матриця зв'язків між нейронами
        input_signals: Вхідні сигнали
        simulation_time: Точки часу симуляції
    
    Повертає:
        Словник з результатами симуляції
    """
    n_neurons = len(neuron_connections)
    
    # Початкові стани нейронів
    neuron_states = np.zeros(n_neurons)
    neuron_states[:len(input_signals)] = input_signals  # встановлення вхідних сигналів
    
    # Історія активності
    activity_history = [neuron_states.copy()]
    
    # Модель нейрона (спрощена)
    def neuron_model(state, inputs, time):
        # Інтеграція вхідних сигналів
        total_input = np.sum(inputs)
        
        # Нелінійна активація (сигмоїда)
        new_state = 1 / (1 + np.exp(-total_input))
        
        # Затухання
        new_state = 0.9 * new_state + 0.1 * state
        
        return new_state
    
    # Симуляція
    dt = simulation_time[1] - simulation_time[0] if len(simulation_time) > 1 else 0.1
    
    for t in simulation_time[1:]:
        new_states = np.zeros(n_neurons)
        
        for i in range(n_neurons):
            # Вхідні сигнали для нейрона i
            inputs = [neuron_connections[i][j] * neuron_states[j] for j in range(n_neurons)]
            new_states[i] = neuron_model(neuron_states[i], inputs, t)
        
        neuron_states = new_states
        activity_history.append(neuron_states.copy())
    
    # Аналіз мережі
    # Середня активність
    mean_activity = np.mean(activity_history)
    
    # Максимальна активність
    max_activity = np.max(activity_history)
    
    # Ступені вузлів
    node_degrees = [np.count_nonzero(row) for row in neuron_connections]
    
    # Коефіцієнт кластеризації
    clustering_coefficients = []
    for i in range(n_neurons):
        neighbors = [j for j, weight in enumerate(neuron_connections[i]) if weight != 0]
        if len(neighbors) < 2:
            clustering_coefficients.append(0)
        else:
            # Кількість з'єднань між сусідами
            edges_between_neighbors = 0
            for j in neighbors:
                for k in neighbors:
                    if j != k and neuron_connections[j][k] != 0:
                        edges_between_neighbors += 1
            # Максимальна кількість можливих з'єднань
            max_edges = len(neighbors) * (len(neighbors) - 1)
            clustering_coefficients.append(edges_between_neighbors / max_edges if max_edges > 0 else 0)
    
    mean_clustering = np.mean(clustering_coefficients)
    
    return {
        'activity_history': [state.tolist() for state in activity_history],
        'mean_activity': float(mean_activity),
        'max_activity': float(max_activity),
        'node_degrees': node_degrees,
        'clustering_coefficients': clustering_coefficients,
        'mean_clustering': float(mean_clustering),
        'n_neurons': n_neurons,
        'simulation_steps': len(simulation_time)
    }

def action_potential_model(membrane_potential: float, 
                         time_span: Tuple[float, float], 
                         n_points: int = 1000) -> Dict[str, Any]:
    """
    Модель потенціалу дії нейрона (модель Ходжкіна-Хакслі, спрощена).
    
    Параметри:
        membrane_potential: Початковий мембранний потенціал (мВ)
        time_span: Інтервал часу (мс)
        n_points: Кількість точок
    
    Повертає:
        Словник з результатами моделювання
    """
    # Часові точки
    t = np.linspace(time_span[0], time_span[1], n_points)
    dt = t[1] - t[0]
    
    # Параметри моделі
    C_m = 1.0  # Ємність мембрани (мкФ/см²)
    g_Na = 120.0  # Провідність Na+ (мСм/см²)
    g_K = 36.0    # Провідність K+ (мСм/см²)
    g_L = 0.3     # Провідність витоку (мСм/см²)
    E_Na = 50.0   # Рівноважний потенціал Na+ (мВ)
    E_K = -77.0   # Рівноважний потенціал K+ (мВ)
    E_L = -54.4   # Рівноважний потенціал витоку (мВ)
    
    # Початкові умови
    V = membrane_potential  # мембранний потенціал
    m = 0.05  # активуючі ворота Na+
    h = 0.6   # інактивуючі ворота Na+
    n = 0.32  # активуючі ворота K+
    
    # Історія
    V_history = [V]
    m_history = [m]
    h_history = [h]
    n_history = [n]
    
    # Стимуляція
    def stimulus(t_val):
        if 1.0 <= t_val <= 1.5:
            return 20.0  # струм стимуляції (мкА/см²)
        else:
            return 0.0
    
    # Симуляція
    for t_val in t[1:]:
        # Струми
        I_Na = g_Na * m**3 * h * (V - E_Na)
        I_K = g_K * n**4 * (V - E_K)
        I_L = g_L * (V - E_L)
        I_stim = stimulus(t_val)
        
        # Зміна мембранного потенціалу
        dV_dt = (I_stim - I_Na - I_K - I_L) / C_m
        V += dV_dt * dt
        
        # Кінетика воріт
        # Активуючі ворота Na+
        alpha_m = 0.1 * (25 - V) / (np.exp((25 - V) / 10) - 1) if V != 25 else 1.0
        beta_m = 4 * np.exp(-V / 18)
        dm_dt = alpha_m * (1 - m) - beta_m * m
        m += dm_dt * dt
        
        # Інактивуючі ворота Na+
        alpha_h = 0.07 * np.exp(-V / 20)
        beta_h = 1 / (np.exp((30 - V) / 10) + 1)
        dh_dt = alpha_h * (1 - h) - beta_h * h
        h += dh_dt * dt
        
        # Активуючі ворота K+
        alpha_n = 0.01 * (10 - V) / (np.exp((10 - V) / 10) - 1) if V != 10 else 0.1
        beta_n = 0.125 * np.exp(-V / 80)
        dn_dt = alpha_n * (1 - n) - beta_n * n
        n += dn_dt * dt
        
        # Зберігання історії
        V_history.append(V)
        m_history.append(m)
        h_history.append(h)
        n_history.append(n)
    
    # Аналіз потенціалу дії
    # Амплітуда
    amplitude = max(V_history) - min(V_history)
    
    # Тривалість
    duration = 0
    threshold = -50.0  # поріг активації
    active = False
    start_time = 0
    
    for i, v in enumerate(V_history):
        if v > threshold and not active:
            active = True
            start_time = t[i]
        elif v <= threshold and active:
            active = False
            duration = t[i] - start_time
            break
    
    # Частота вогнів (якщо є декілька спайків)
    spike_times = []
    for i in range(1, len(V_history)-1):
        if V_history[i] > V_history[i-1] and V_history[i] > V_history[i+1] and V_history[i] > 0:
            spike_times.append(t[i])
    
    firing_rate = len(spike_times) / (time_span[1] - time_span[0]) * 1000 if len(spike_times) > 1 else 0
    
    return {
        'membrane_potential': V_history,
        'sodium_activation': m_history,
        'sodium_inactivation': h_history,
        'potassium_activation': n_history,
        'time_points': t.tolist(),
        'amplitude': amplitude,
        'duration': duration,
        'spike_times': spike_times,
        'firing_rate': firing_rate
    }

if __name__ == "__main__":
    # Тестування функцій модуля
    print("Тестування модуля обчислювальної біології PyNexus")
    
    # Тест аналізу послідовності ДНК
    dna_seq = "ATGCGATCGTAGCTAG"
    dna_analysis = dna_sequence_analysis(dna_seq)
    print(f"GC-вміст: {dna_analysis['gc_content']:.2f}")
    print(f"Довжина послідовності: {dna_analysis['length']}")
    
    # Тест вирівнювання послідовностей
    seq1 = "ACGTACGT"
    seq2 = "ACGTACGT"
    alignment = sequence_alignment(seq1, seq2)
    print(f"Оцінка вирівнювання: {alignment['alignment_score']}")
    print(f"Ідентичність: {alignment['identity']:.2f}")
    
    # Тест аналізу білка
    protein_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    protein_analysis = protein_structure_analysis(protein_seq)
    print(f"Молекулярна маса: {protein_analysis['molecular_weight']:.2f}")
    print(f"Гідрофобність: {protein_analysis['hydrophobicity']:.2f}")
    
    # Тест симуляції згортання білка
    folding_result = protein_folding_simulation("ACDEFGHIKLMNPQRSTVWY", n_steps=100)
    print(f"Енергія згортання: {folding_result['final_energy']:.2f}")
    print(f"Кількість контактів: {folding_result['contacts']}")
    
    # Тест популяційної генетики
    pop_gen_result = population_genetics_simulation([0.7, 0.3], [1.0, 0.9, 0.8], 1000, 100)
    print(f"Рівноважна частота A: {pop_gen_result['equilibrium_p']:.3f}")
    print(f"Покоління до рівноваги: {pop_gen_result['equilibrium_generation']}")
    
    print("Тестування завершено!")