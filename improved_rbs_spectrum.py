import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import periodictable as pt
import pubchempy as pcp
import re

# Experiment parameters
E0 = 4  # MeV (incident proton energy)
theta = 170  # Scattering angle in degrees
thickness_nm = 150  # target layer thickness in nm
resolution_keV = 15  # Detector resolution (keV)
#substrate_thickness_nm = 2000

# target and substrate selection
target_material = "Au"
substrate_material = "Si"
ion_symbol = "H" # "H" for proton or "He" for "alpha" particles

#---retrieve ion data from periodictable pkg---#
ion = pt.elements.symbol(ion_symbol)
M_ion = ion.mass
Z_ion = ion.number
name_ion = ion.name
print(f"Incident ion is: {ion}")


def get_material_properties(material_name, material_type=""):
    """
    ดึงข้อมูลคุณสมบัติของวัสดุจาก periodictable และ pubchempy
    
    Parameters:
    - material_name: ชื่อวัสดุ (เช่น "Au", "Si", "SiO2", "glass")
    - material_type: ประเภทของวัสดุในการทดลอง (เช่น "substrate", "target", "film")
                     ใช้เพื่อการบันทึกและการแสดงผลเท่านั้น
    
    Returns:
    - dictionary ที่มีคุณสมบัติของวัสดุ
    """
    # elements or compound check
    try:
        # พยายามดึงข้อมูลจาก periodictable ก่อน (สำหรับธาตุ)
        element = pt.elements.symbol(material_name)
        Z_substrate = element.number
        M_substrate = element.mass
        density_substrate = element.density  # g/cm³
        # แปลงหน่วยความหนาแน่นเป็น atoms/cm³
        avogadro = 6.022e23  # Avogadro's number
        density_substrate_atoms_per_cm3 = density_substrate * avogadro / M_substrate
        substrate_name = element.symbol
        
        return {
            'Z': Z_substrate,
            'M': M_substrate,
            'density': density_substrate,
            'density_atoms_per_cm3': density_substrate_atoms_per_cm3,
            'name': substrate_name,
            'type': 'element'
        }
        
    except (ValueError, AttributeError):
        # ถ้าไม่ใช่ธาตุ ให้ลองค้นหาในฐานข้อมูล PubChem (สำหรับสารประกอบ)
        try:
            compounds = pcp.get_compounds(material_name, 'name')
            if compounds:
                compound = compounds[0]
                # ดึงข้อมูลสารประกอบ
                formula = compound.molecular_formula
                molecular_weight = compound.molecular_weight
                
                # คำนวณ effective Z (ประมาณการ)
                # นับจำนวนอะตอมในสูตรเคมี
                element_counts = {}
                current_element = ""
                current_count = ""
                
                for char in formula:
                    if char.isupper():
                        if current_element:
                            element_counts[current_element] = int(current_count) if current_count else 1
                        current_element = char
                        current_count = ""
                    elif char.islower():
                        current_element += char
                    elif char.isdigit():
                        current_count += char
                
                # เพิ่มอะตอมสุดท้าย
                if current_element:
                    element_counts[current_element] = int(current_count) if current_count else 1
                
                # คำนวณ effective Z และ effective M
                effective_Z = 0
                total_atoms = 0
                
                for elem_symbol, count in element_counts.items():
                    element = pt.elements.symbol(elem_symbol)
                    effective_Z += element.number * count
                    total_atoms += count
                
                # คำนวณ Z เฉลี่ย
                average_Z = effective_Z / total_atoms
                
                # ค่าความหนาแน่นถ้ามี (บางครั้ง PubChem อาจไม่มีข้อมูลนี้)
                # ในกรณีนี้อาจต้องใช้ค่าประมาณหรือค่าเริ่มต้น
                density = getattr(compound, 'density', 2.0)  # g/cm³ (ค่าเริ่มต้นถ้าไม่มีข้อมูล)
                
                # แปลงหน่วยความหนาแน่นเป็น atoms/cm³
                avogadro = 6.022e23
                density_atoms_per_cm3 = density * avogadro / molecular_weight
                
                return {
                    'Z': average_Z,
                    'M': molecular_weight,
                    'density': density,
                    'density_atoms_per_cm3': density_atoms_per_cm3,
                    'name': substrate_material,
                    'type': 'compound'
                }
            else:
                raise ValueError(f"{substrate_material} data not found!!!")
                
        except Exception as e:
            # ถ้าไม่พบทั้งในฐานข้อมูลธาตุและสารประกอบ ให้ใช้ค่าเริ่มต้นหรือข้อมูลที่กำหนดเอง
            # ฐานข้อมูลวัสดุที่กำหนดไว้
            predefined_materials = {
                "glass": {
                    'Z': 10,  # ค่าประมาณ effective Z
                    'M': 60.1,  # ค่าประมาณมวลโมเลกุลเฉลี่ย (SiO2 หลัก)
                    'density': 2.5,  # g/cm³
                    'density_atoms_per_cm3': 2.5 * 6.022e23 / 60.1,
                    'name': 'Glass',
                    'type': 'material',
                    'role': material_type
                },
                "graphite": {
                    'Z': 6,  # คาร์บอน
                    'M': 12.01,
                    'density': 2.2,  # g/cm³
                    'density_atoms_per_cm3': 2.2 * 6.022e23 / 12.01,
                    'name': 'Graphite',
                    'type': 'material',
                    'role': material_type
                },
                "aluminum oxide": {
                    'Z': 13,  # ค่าประมาณ (Al มี Z=13)
                    'M': 101.96,  # Al2O3
                    'density': 3.95,  # g/cm³
                    'density_atoms_per_cm3': 3.95 * 6.022e23 / 101.96,
                    'name': 'Al2O3',
                    'type': 'material',
                    'role': material_type
                },
                "quartz": {
                    'Z': 10,  # ค่าประมาณ (Si มี Z=14, O มี Z=8)
                    'M': 60.08,  # SiO2
                    'density': 2.65,  # g/cm³
                    'density_atoms_per_cm3': 2.65 * 6.022e23 / 60.08,
                    'name': 'Quartz',
                    'type': 'material',
                    'role': material_type
                }
            }
            
            # คำนวณ default material (ถ้าไม่พบในฐานข้อมูล)
            default_material = {
                'Z': 14,  # ใช้ค่า Si เป็นค่าเริ่มต้น
                'M': 28.09,
                'density': 2.33,  # g/cm³
                'density_atoms_per_cm3': 2.33 * 6.022e23 / 28.09,
                'name': material_name,
                'type': 'unknown',
                'role': material_type
            }
            
            # ตรวจสอบว่ามีในฐานข้อมูลที่กำหนดไว้หรือไม่
            for key, material in predefined_materials.items():
                if material_name.lower() == key.lower():
                    return material
            
            print(f"Warning: {material_name} data not found!!!, using default")
            return default_material


target_props = get_material_properties(target_material, "target")
substrate_props = get_material_properties(substrate_material, "substrate")

Z_target = target_props['Z']
M_target = target_props['M']
density_target = target_props['density']
density_target_atoms_per_cm3 = target_props['density_atoms_per_cm3']
target_name = target_props['name']


Z_substrate = substrate_props['Z']
M_substrate = substrate_props['M']
density_substrate = substrate_props['density']
density_substrate_atoms_per_cm3 = substrate_props['density_atoms_per_cm3']
substrate_name = substrate_props['name']

'''
# กรณีมีวัสดุหลายชั้น
film_layers = [
    {"material": "Au", "thickness": 100},  # ชั้นทอง 100 nm
    {"material": "Ti", "thickness": 10},   # ชั้นไทเทเนียม 10 nm
]

# ดึงข้อมูลวัสดุสำหรับแต่ละชั้น
for i, layer in enumerate(film_layers):
    layer_props = get_material_properties(layer["material"], f"layer_{i+1}")
    layer.update(layer_props)
    
    print(f"Layer {i+1}: {layer['name']}, thickness={layer['thickness']} nm")
    print(f"  Z={layer['Z']}, M={layer['M']}, density={layer['density']} g/cm³")
'''


capitalized_name_ion = name_ion.capitalize()
print(capitalized_name_ion)
# Load SRIM data as strings (to handle units)
with open(f"D:/Programs/SRIM/SRIM Outputs/{capitalized_name_ion} in {target_name}.txt", "r") as file:
    lines = file.readlines()[23:-13]  # Skip header (first 23 rows, last 13 rows)

# Extract energy and projected range columns
data = [line.split() for line in lines if len(line.split()) >= 5]  # Ensure enough columns
#Kinetic_energies = np.array([float(row[0]) for row in data])  # Convert energy column to float
Kinetic_energies = []

for row in data:
    value = float(row[0])
    unit = row[1].strip()  # using strip() to delete unnecessary space

    # แสดงข้อมูลเพื่อตรวจสอบ
    #print(f"Processing row: {row[0]} {unit}")

    if unit == "keV":
        value /= 1000
    Kinetic_energies.append(value)

Kinetic_energies = np.array(Kinetic_energies)

#ranges = np.array([float(re.findall(r"[\d.]+", row[4])[0]) for row in data])  # Extract numeric part
ranges = np.array([
    float(re.findall(r"[\d.]+", row[4])[0]) /10000 if row[4].endswith("A") else float(re.findall(r"[\d.]+", row[4])[0])
    for row in data
])

# Find projected range corresponding to proton energy
projected_range_um_srim = ranges[np.isclose(Kinetic_energies, E0)][0]

print(f"projected range from SRIM: {projected_range_um_srim} um")

# Corrected stopping power using SRIM range (58.19 µm for 5 MeV protons in gold)
#projected_range_um_srim = 0.8592 #58.19  # Corrected from SRIM
stopping_power_mev_per_um_srim = E0 / projected_range_um_srim  # MeV per µm


# Improved physics functions
def kinematic_factor(M1, M2, theta_deg):
    """Calculate kinematic factor for elastic scattering."""
    theta_rad = np.radians(theta_deg)
    return ((M1 * np.cos(theta_rad) + np.sqrt(M2**2 - M1**2 * np.sin(theta_rad)**2)) / (M1 + M2))**2

def energy_loss_vs_depth(initial_energy, depth_nm, stopping_power):
    """Calculate energy after traveling through material."""
    # Convert nm to um for calculation
    depth_um = depth_nm / 1000
    # Calculate energy loss
    energy_loss = stopping_power * depth_um
    return initial_energy - energy_loss

def rutherford_cross_section(Z1, Z2, E, theta_deg):
    """Calculate Rutherford cross-section."""
    theta_rad = np.radians(theta_deg)
    constant = 1.296  # (e^2/4πε0)^2 in MeV^2·fm^2
    cross_section = (Z1 * Z2 * constant / (4 * E))**2 * (1 / np.sin(theta_rad/2)**4)
    return cross_section

def bohr_straggling(Z1, Z2, thickness_nm):
    """Calculate energy straggling using Bohr formula."""
    # Z1, Z2: atomic numbers
    # thickness_nm: target thickness in nm
    # Returns energy straggling in keV
    constant = 0.157  # Bohr straggling constant
    thickness_cm = thickness_nm * 1e-7
    straggling_keV = constant * Z1 * Z2 * np.sqrt(thickness_cm)
    return straggling_keV

def add_pileup_effect(spectrum, energies, probability=0.05):
    """Simulate pile-up effect in detector."""
    # Create a simplified convolution of spectrum with itself
    pileup = np.zeros_like(spectrum)
    energy_step = energies[1] - energies[0]
    
    for i in range(len(energies)):
        for j in range(len(energies)):
            e_sum = energies[i] + energies[j]
            if e_sum <= max(energies):
                idx = int((e_sum - min(energies)) / energy_step)
                if 0 <= idx < len(pileup):
                    pileup[idx] += probability * spectrum[i] * spectrum[j] / np.max(spectrum)
    
    return spectrum + pileup

# Calculate kinematic factors
K_target = kinematic_factor(M_ion, M_target, theta)
K_substrate = kinematic_factor(M_ion, M_substrate, theta)

# Energy peaks
E_target_surface = E0 * K_target
E_substrate_surface = E0 * K_substrate

# Energy range for spectrum
energies = np.linspace(E0-(E0*0.8), E0+(E0*0.02), 500)  # Energy range (MeV)
#energies = np.linspace(0.08, 0.51, 500)
# Calculate energy spectrum from target layer considering depth
depths = np.linspace(0, thickness_nm, 50)

# Target spectrum calculation
def calculate_target_spectrum(energies, depths, E0, stopping_power, K_factor, Z_projectile, Z_target, theta):
    # create array for store the spectrum data
    spectrum = np.zeros(len(energies))
    
    # คำนวณช่วงพลังงาน
    energy_step = energies[1] - energies[0]
    
    # ความหนาแน่นของ target (atoms/cm^3)
    density = density_target_atoms_per_cm3
    
    # ความหนาของชั้นที่พิจารณา (cm)
    delta_depth_cm = (depths[1] - depths[0]) * 1.0e-7  # nm -> cm
    
    # คำนวณ straggling parameter
    straggling_factor = bohr_straggling(Z_projectile, Z_target, thickness_nm)
    
    # คำนวณแบบ slice-by-slice
    for i, depth in enumerate(depths):
        # พลังงานขาเข้า
        energy_in = energy_loss_vs_depth(E0, depth, stopping_power)
        
        # พลังงานหลังกระเจิง
        energy_out = energy_in * K_factor
        
        # พลังงานขาออก (ต้องคำนวณระยะทางแนวเฉียง)
        exit_angle_correction = 1.0 / np.cos(np.radians(180 - theta))
        exit_depth = depth * exit_angle_correction
        energy_final = energy_loss_vs_depth(energy_out, exit_depth, stopping_power)
        
        # คำนวณ cross-section (ขึ้นกับพลังงานขณะชน)
        sigma = rutherford_cross_section(Z_projectile, Z_target, energy_in, theta)
        
        # คำนวณจำนวนนับที่จะได้จากชั้นนี้ (ตามทฤษฎี RBS)
        # dY = σ(E) × Ω × N × t
        # โดย Y = yield, σ = cross-section, Ω = solid angle (คงที่), N = atomic density, t = thickness
        yield_factor = sigma * density * delta_depth_cm
        
        # ใส่จำนวนนับนี้ลงในช่วงพลังงานที่เหมาะสม โดยพิจารณา energy straggling
        # สำหรับแต่ละชั้นความลึกจะมีการกระจายพลังงานแบบ Gaussian
        
        # คำนวณความกว้างของ Gaussian จาก straggling และ detector resolution
        width_total = np.sqrt((resolution_keV/1000)**2 + (straggling_factor/1000)**2)
        
        # นำจำนวนนับไปกระจายในสเปกตรัมตามการกระจายพลังงาน
        for j, e in enumerate(energies):
            energy_diff = e - energy_final
            # การกระจายแบบ Gaussian ที่มีความไม่สมมาตร (เอียงไปทางพลังงานต่ำ)
            if energy_diff <= 0:
                # ทางด้านพลังงานต่ำ (tail ยาวกว่า)
                gaussian = yield_factor * np.exp(-(energy_diff**2) / (2.2 * width_total**2))
            else:
                # ทางด้านพลังงานสูง (tail สั้นกว่า)
                gaussian = yield_factor * np.exp(-(energy_diff**2) / (1.8 * width_total**2))
            
            spectrum[j] += gaussian
    
    # ปรับสเกลให้อยู่ในช่วงที่เหมาะสม
    return spectrum / np.max(spectrum) * 500  # ปรับให้มีความสูงประมาณ 500 counts

# ใช้ฟังก์ชันนี้แทนการสร้าง target_spectrum ด้วย zeros_like
target_spectrum = calculate_target_spectrum(
    energies, 
    depths, 
    E0, 
    stopping_power_mev_per_um_srim, 
    K_target, 
    Z_ion, 
    Z_target, 
    theta
)

# Substrate spectrum calculation
def calculate_substrate_spectrum(energies, E0, thickness_target_nm, stopping_power_target, 
                                K_substrate, Z_ion, Z_substrate, theta):
    # พลังงานหลังผ่านชั้น target
    energy_after_target = energy_loss_vs_depth(E0, thickness_target_nm, stopping_power_target)
    
    # พลังงานหลังกระเจิงจากวัสดุฐานรอง
    energy_after_scattering = energy_after_target * K_substrate
    
    # พลังงานหลังผ่านชั้นทองขาออก
    exit_angle_correction = 1.0 / np.cos(np.radians(180 - theta))
    energy_final = energy_loss_vs_depth(energy_after_scattering, thickness_target_nm * exit_angle_correction, stopping_power_target)
    
    # คำนวณ cross-section
    sigma_substrate = rutherford_cross_section(Z_ion, Z_substrate, energy_after_target, theta)
    
    # คำนวณความกว้างของพีค (detector resolution + ผลจากการผ่านชั้นทอง)
    straggling_target_layer = bohr_straggling(Z_ion, Z_target, thickness_target_nm)
    width_total_substrate = np.sqrt((resolution_keV/1000)**2 + (straggling_target_layer/1000)**2) * 1.5  # กว้างขึ้นเนื่องจากผ่านชั้น target 2 ครั้ง
    
    # สร้างพีคฐานรองที่มีรูปร่างเหมือนจริง (asymmetric)
    substrate_spectrum = np.zeros_like(energies)
    for i, e in enumerate(energies):
        energy_diff = e - energy_final
        # พีคที่มีความไม่สมมาตร (เอียงไปทางพลังงานต่ำ)
        if energy_diff <= 0:
            # ทางด้านพลังงานต่ำ (tail ยาวมาก)
            substrate_spectrum[i] = sigma_substrate * np.exp(-(energy_diff**2) / (3.0 * width_total_substrate**2))
        else:
            # ทางด้านพลังงานสูง (tail สั้น)
            substrate_spectrum[i] = sigma_substrate * np.exp(-(energy_diff**2) / (1.2 * width_total_substrate**2))
    
    # ปรับความสูงของพีคตามหมายเลขอะตอม (Z^2 dependency)
    height_scale = (Z_substrate / 14)**2 * 200
    
    # ปรับสเกลให้อยู่ในช่วงที่เหมาะสม
    return substrate_spectrum / np.max(substrate_spectrum) * height_scale  # ปรับความสูงตามหมายเลขอะตอม

substrate_spectrum = calculate_substrate_spectrum(
    energies, 
    E0, 
    thickness_nm, 
    stopping_power_mev_per_um_srim, 
    K_substrate, 
    Z_ion, 
    Z_substrate, 
    theta
)

# เพิ่มลักษณะพื้นหลังที่มีในการทดลองจริง
def add_realistic_background(energies, min_counts=2, slope=0.5):
    # สร้างพื้นหลังที่มีระดับต่ำแต่เพิ่มขึ้นเล็กน้อยที่พลังงานต่ำ
    background = min_counts + slope * (max(energies) - energies)
    return background

# เพิ่มสัญญาณรบกวนและพื้นหลังที่สมจริง
background = add_realistic_background(energies)
combined_spectrum = target_spectrum + substrate_spectrum
spectrum_with_noise = combined_spectrum + background + np.random.poisson(3, size=energies.shape)

# ฟังก์ชันที่ปรับปรุงสำหรับการจำลอง pile-up ที่สมจริงมากขึ้น
def improved_pileup(spectrum, energies, detector_deadtime=0.05):
    # สร้างสเปกตรัม pile-up โดยใช้การคอนโวลูชั่นที่มีการถ่วงน้ำหนัก
    pileup = np.zeros_like(spectrum)
    energy_step = energies[1] - energies[0]
    max_counts = np.max(spectrum)
    
    # คำนวณโอกาสที่จะเกิด pile-up ตามสัดส่วนของจำนวนนับ
    pileup_probability = detector_deadtime * (spectrum / max_counts)
    
    # สร้าง pile-up โดยการคอนโวลูชั่นที่มีการถ่วงน้ำหนัก
    for i, e1 in enumerate(energies):
        if spectrum[i] <= 0:
            continue
            
        for j, e2 in enumerate(energies):
            if spectrum[j] <= 0:
                continue
                
            # พลังงานรวม
            e_sum = e1 + e2
            if e_sum <= max(energies):
                # หาตำแหน่งที่จะใส่จำนวนนับลงในสเปกตรัม
                idx = int((e_sum - min(energies)) / energy_step)
                if 0 <= idx < len(pileup):
                    # คำนวณจำนวนนับที่จะเพิ่มโดยถ่วงน้ำหนักตามโอกาสในการเกิด pile-up
                    contribution = pileup_probability[i] * spectrum[i] * pileup_probability[j] * spectrum[j] / max_counts
                    pileup[idx] += contribution
    
    return spectrum + 0.5 * pileup  # ลดน้ำหนักของ pile-up ลงเล็กน้อย

# Use the improved pile-up function
final_spectrum = improved_pileup(spectrum_with_noise, energies, detector_deadtime=0.02)

# REPLACEMENT SECTION END

# Ensure no negative values
final_spectrum = np.maximum(final_spectrum, 0)

# Create dataset
rbs_data = pd.DataFrame({"Energy (MeV)": energies, "Counts": final_spectrum})

# Save as CSV file
rbs_data.to_csv(f"improved_rbs_simulation_{ion}_{E0}MeV_{target_name}{thickness_nm}nm_on_{substrate_name}.csv", index=False)

# Calculate energy after passing through entire target layer for interface marking
energy_at_substrate = energy_loss_vs_depth(E0, thickness_nm, stopping_power_mev_per_um_srim)
E_substrate_interface = energy_at_substrate * K_substrate
# Energy after passing back through target layer
E_substrate_detected = energy_loss_vs_depth(E_substrate_interface, thickness_nm, stopping_power_mev_per_um_srim)

# Plot the RBS spectrum
plt.figure(figsize=(10, 6))
plt.semilogy(rbs_data["Energy (MeV)"], rbs_data["Counts"], label="Simulated RBS Spectrum", color="blue")
plt.fill_between(rbs_data["Energy (MeV)"], rbs_data["Counts"], alpha=0.3, color="lightblue")
plt.axvline(x=E_target_surface, color='red', linestyle="--", label=f"{target_name} Surface (~{E_target_surface:.2f} MeV)")
plt.axvline(x=E_substrate_detected, color='green', linestyle="--", 
            label=f"{substrate_name} Interface (~{E_substrate_detected:.2f} MeV)")

# Calculate target-substrate interface energy based on deepest target layer
min_target_energy = min([energy_loss_vs_depth(energy_in * K_target, depth, stopping_power_mev_per_um_srim) 
                      for depth, energy_in in zip(depths, [energy_loss_vs_depth(E0, d, stopping_power_mev_per_um_srim) for d in depths])])
plt.axvline(x=min_target_energy, color='orange', linestyle="--", 
            label=f"{target_name}-{substrate_name} Interface Energy (~{min_target_energy:.2f} MeV)")

plt.xlabel("Energy (MeV)")
plt.ylabel("Counts (log scale)")
plt.title(f"Improved RBS Simulation ({E0} MeV {name_ion} ion on {thickness_nm} nm {target_name}/{substrate_name})")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xlim(min(energies), max(energies))

# Add a second plot showing linear scale
plt.figure(figsize=(10, 6))
plt.plot(rbs_data["Energy (MeV)"], rbs_data["Counts"], label="Simulated RBS Spectrum", color="blue")
plt.axvline(x=E_target_surface, color='red', linestyle="--", label=f"{target_name} Surface (~{E_target_surface:.2f} MeV)")
plt.axvline(x=E_substrate_detected, color='green', linestyle="--", label=f"{substrate_name} Interface (~{E_substrate_detected:.2f} MeV)")
plt.axvline(x=min_target_energy, color='orange', linestyle="--", label=f"{target_name}-{substrate_name} Interface Energy (~{min_target_energy:.2f} MeV)")

plt.xlabel("Energy (MeV)")
plt.ylabel("Counts")
plt.title("Improved RBS Simulation (Linear Scale)")
plt.legend()
plt.grid(True)
plt.xlim(min(energies), max(energies))

# Add annotation explaining the physics
annotation_text = (
    "Physics Included:\n"
    "- Kinematic factor calculations\n"
    "- Energy loss vs. depth\n"
    "- Rutherford cross-section\n"
    "- Bohr energy straggling\n"
    "- Pile-up effects\n"
    "- Detector resolution\n"
    "- Asymmetric peak shapes\n"
    "- Realistic background"
)
plt.annotate(annotation_text, xy=(0.02, 0.97), xycoords='axes fraction', 
            va='top', bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

plt.tight_layout()
plt.show()

# Print summary of simulation parameters
print("\nSimulation Parameters Summary:")
print(f"Incident {capitalized_name_ion} Energy: {E0} MeV")
print(f"Scattering Angle: {theta} degrees")
print(f"{target_name} Layer Thickness: {thickness_nm} nm")
print(f"Detector Resolution: {resolution_keV} keV")
print(f"{target_name} Surface Energy: {E_target_surface:.2f} MeV")
print(f"{substrate_name} Interface Energy: {E_substrate_detected:.2f} MeV")
print(f"Stopping Power (from SRIM): {stopping_power_mev_per_um_srim:.3f} MeV/μm")
straggling_target = bohr_straggling(Z_ion, Z_target, thickness_nm)
print(f"Energy Straggling in {target_name} Layer: {straggling_target:.2f} keV")