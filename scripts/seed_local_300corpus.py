# scripts/seed_local_300.py
"""
Seed Local Permanent Layer with 300 essential medical facts.
Optimized for 5K cloud corpus (30% of 1K local capacity).

Author: Saberzerker
Date: 2025-11-16
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from src.config import BASE_LAYER_PATH, VECTOR_DIMENSION


# 300 ESSENTIAL MEDICAL FACTS (Core knowledge for offline operation)
PERMANENT_FACTS_300 = [
    # CARDIOVASCULAR (40 facts)
    "Hypertension is blood pressure above 140/90 mmHg, major risk factor for heart disease and stroke.",
    "Stage 1 hypertension is 130-139/80-89 mmHg, often manageable with lifestyle changes alone.",
    "Stage 2 hypertension is 140+/90+ mmHg, typically requires medication and lifestyle changes.",
    "Hypertensive crisis is BP above 180/120 mmHg, requiring immediate emergency medical attention.",
    "Atherosclerosis is plaque buildup in arteries, restricting blood flow to vital organs.",
    "Coronary artery disease develops when heart arteries become narrowed by cholesterol plaque.",
    "Myocardial infarction occurs when coronary blockage prevents oxygen reaching heart muscle.",
    "Heart attack symptoms include chest pressure, arm pain, shortness of breath, and sweating.",
    "Women may have atypical heart attack symptoms including nausea, fatigue, and back pain.",
    "Angina is chest pain from reduced heart blood flow, typically triggered by exertion or stress.",
    "Stable angina follows predictable patterns while unstable angina occurs at rest.",
    "Atrial fibrillation is irregular heartbeat increasing stroke risk through blood clot formation.",
    "Heart failure occurs when heart cannot pump enough blood to meet body's metabolic needs.",
    "Congestive heart failure causes fluid accumulation in lungs, legs, and abdomen.",
    "ACE inhibitors treat hypertension and heart failure by reducing angiotensin II formation.",
    "Beta-blockers slow heart rate and reduce blood pressure by blocking epinephrine effects.",
    "Statins lower cholesterol by blocking HMG-CoA reductase enzyme in liver.",
    "Aspirin therapy reduces heart attack risk by preventing platelet aggregation and clotting.",
    "Cardiac catheterization visualizes coronary arteries and can open blockages with stents.",
    "Coronary artery bypass grafting redirects blood flow around blocked coronary arteries.",
    "Electrocardiogram records heart's electrical activity, detecting rhythm abnormalities and ischemia.",
    "Echocardiogram uses ultrasound to visualize heart structure and assess pumping function.",
    "Stress test evaluates heart function during exercise, revealing ischemia not apparent at rest.",
    "LDL cholesterol below 100 mg/dL is optimal, while HDL above 60 mg/dL protects heart.",
    "Smoking cessation dramatically reduces cardiovascular disease risk within months of quitting.",
    "Mediterranean diet reduces heart disease risk through healthy fats and plant-based foods.",
    "Regular aerobic exercise strengthens heart and reduces cardiovascular disease risk by 30-40%.",
    "Stroke occurs when brain blood supply is interrupted, causing rapid brain cell death.",
    "Ischemic stroke is caused by blood clot blocking brain artery, treated with clot-busting drugs.",
    "Hemorrhagic stroke involves blood vessel rupture, creating pressure damaging brain tissue.",
    "FAST acronym for stroke: Face drooping, Arm weakness, Speech difficulty, Time to call 911.",
    "tPA dissolves clots if given within 4.5 hours of ischemic stroke symptom onset.",
    "Deep vein thrombosis forms blood clots in deep leg veins, risking pulmonary embolism.",
    "Pulmonary embolism occurs when blood clot travels to lungs, potentially life-threatening.",
    "Peripheral artery disease causes leg pain with walking from arterial narrowing.",
    "Aortic aneurysm is dangerous artery bulging that can rupture, causing internal bleeding.",
    "Pacemaker regulates heart rhythm by delivering electrical impulses when heart beats too slowly.",
    "Implantable defibrillator detects and corrects dangerous arrhythmias automatically.",
    "Cardiac rehabilitation combines exercise, education, and support improving heart disease outcomes.",
    "Blood pressure control is single most important intervention for preventing stroke.",
    
    # DIABETES & ENDOCRINE (40 facts)
    "Type 1 diabetes is autoimmune destruction of insulin-producing pancreatic beta cells.",
    "Type 2 diabetes involves insulin resistance where body cells don't respond properly to insulin.",
    "Prediabetes is blood glucose 100-125 mg/dL fasting, indicating high diabetes risk.",
    "Gestational diabetes develops during pregnancy, increasing risks for mother and baby.",
    "Hemoglobin A1C measures average blood glucose over previous 2-3 months.",
    "A1C target for most diabetics is below 7% to prevent long-term complications.",
    "Fasting glucose above 126 mg/dL on two occasions confirms diabetes diagnosis.",
    "Insulin regulates blood sugar by facilitating glucose uptake into body cells.",
    "Metformin is first-line type 2 diabetes medication, reducing liver glucose production.",
    "GLP-1 agonists improve glucose control, promote weight loss, and reduce cardiovascular events.",
    "SGLT2 inhibitors increase urinary glucose excretion, lowering blood sugar and protecting heart.",
    "Continuous glucose monitors provide real-time glucose readings, improving diabetes management.",
    "Diabetic ketoacidosis is life-threatening type 1 complication requiring emergency treatment.",
    "Hypoglycemia below 70 mg/dL causes shakiness, confusion, and can lead to unconsciousness.",
    "Diabetic neuropathy damages nerves causing pain, numbness, especially in feet.",
    "Diabetic retinopathy damages retinal blood vessels, leading to vision loss if untreated.",
    "Diabetic nephropathy causes kidney damage, potentially progressing to kidney failure.",
    "Diabetes increases cardiovascular disease risk 2-4 times through accelerated atherosclerosis.",
    "Foot care is critical for diabetics to prevent ulcers, infections, and amputations.",
    "Carbohydrate counting helps diabetics match insulin doses to food intake for better control.",
    "Regular exercise improves insulin sensitivity and glucose control in both diabetes types.",
    "Weight loss of 5-10% significantly improves type 2 diabetes control and may achieve remission.",
    "Hypothyroidism slows metabolism, causing fatigue, weight gain, and cold intolerance.",
    "Hyperthyroidism accelerates metabolism, causing weight loss, heat intolerance, and anxiety.",
    "TSH test screens thyroid function, elevated in hypothyroidism, suppressed in hyperthyroidism.",
    "Levothyroxine replaces thyroid hormone in hypothyroidism, taken daily on empty stomach.",
    "Hashimoto's thyroiditis is autoimmune hypothyroidism, most common thyroid dysfunction cause.",
    "Graves' disease is autoimmune hyperthyroidism, often causing bulging eyes.",
    "Cushing's syndrome results from excess cortisol, causing weight gain and hypertension.",
    "Addison's disease involves insufficient cortisol, causing fatigue and low blood pressure.",
    "Polycystic ovary syndrome causes hormonal imbalance, irregular periods, and infertility.",
    "Growth hormone deficiency in children impairs growth, requiring hormone replacement therapy.",
    "Metabolic syndrome combines obesity, hypertension, high triglycerides, and insulin resistance.",
    "Thyroid nodules are common, usually benign, but require evaluation to exclude cancer.",
    "Radioactive iodine treats hyperthyroidism by selectively destroying overactive thyroid tissue.",
    "Adrenal insufficiency requires immediate steroid replacement during illness to prevent crisis.",
    "Diabetes management requires monitoring blood glucose, medications, diet, and exercise.",
    "Bariatric surgery can induce type 2 diabetes remission through metabolic changes.",
    "Insulin pump delivers continuous insulin, closely mimicking normal pancreatic function.",
    "Glycemic index ranks carbohydrates by blood sugar impact, guiding diabetic food choices.",
    
    # RESPIRATORY (30 facts)
    "Asthma is chronic airway inflammation causing reversible bronchoconstriction and wheezing.",
    "Asthma affects 25 million Americans, leading chronic disease in children.",
    "Asthma triggers include allergens, exercise, cold air, smoke, and viral infections.",
    "Quick-relief bronchodilators rapidly open airways during asthma attacks.",
    "Inhaled corticosteroids control asthma inflammation long-term, reducing attack frequency.",
    "Peak flow meter measures airflow obstruction, helping monitor asthma control.",
    "COPD combines chronic bronchitis and emphysema, causing progressive breathing difficulty.",
    "Smoking causes 80-90% of COPD cases through chronic airway irritation and damage.",
    "COPD symptoms include chronic cough, sputum production, and progressive shortness of breath.",
    "Spirometry diagnoses COPD by measuring airflow obstruction severity through FEV1/FVC ratio.",
    "Smoking cessation is most important COPD intervention, significantly slowing disease progression.",
    "Bronchodilators relieve COPD symptoms by relaxing airway muscles and improving breathing.",
    "Pulmonary rehabilitation combines exercise, education, and support improving COPD outcomes.",
    "Supplemental oxygen therapy increases survival in COPD patients with chronic hypoxemia.",
    "Pneumonia inflames lung air sacs, filling them with fluid impairing gas exchange.",
    "Bacterial pneumonia often follows viral respiratory infection, causing productive cough and fever.",
    "Antibiotic choice for pneumonia depends on likely pathogens, severity, and resistance patterns.",
    "Pneumococcal vaccine prevents Streptococcus pneumoniae, leading bacterial pneumonia cause.",
    "Tuberculosis is bacterial infection primarily affecting lungs, requiring prolonged antibiotic treatment.",
    "TB symptoms include chronic cough, night sweats, weight loss, and blood-tinged sputum.",
    "Active TB requires 6-9 months of multiple antibiotics to prevent resistance and ensure cure.",
    "Pulmonary embolism blocks pulmonary arteries with blood clots, causing sudden dyspnea.",
    "Lung cancer is leading cancer killer, with smoking responsible for 85% of cases.",
    "Low-dose CT screening reduces lung cancer mortality in high-risk current and former smokers.",
    "Bronchitis inflames bronchial tubes, causing cough and mucus production, often viral.",
    "Sleep apnea causes repeated breathing interruptions during sleep, increasing cardiovascular risk.",
    "CPAP therapy maintains positive airway pressure, preventing airway collapse in sleep apnea.",
    "Influenza causes seasonal respiratory illness, preventable with annual vaccination.",
    "COVID-19 vaccines dramatically reduce severe disease, hospitalization, and death risk.",
    "Inhaler technique is critical for asthma and COPD control, requires proper patient education.",
    
    # NEUROLOGICAL (30 facts)
    "Alzheimer's disease progressively destroys memory through brain cell death and protein accumulation.",
    "Beta-amyloid plaques and tau tangles are Alzheimer's hallmark brain pathological changes.",
    "Alzheimer's affects 6.7 million Americans aged 65+, projected to triple by 2060.",
    "Early Alzheimer's symptoms include memory loss, confusion, and difficulty with familiar tasks.",
    "Cholinesterase inhibitors modestly improve Alzheimer's cognitive symptoms and daily function.",
    "Memantine regulates glutamate activity in moderate-to-severe Alzheimer's disease.",
    "Parkinson's disease causes dopamine deficiency in brain, producing motor symptoms.",
    "Parkinson's classic triad: resting tremor, rigidity, and bradykinesia (slow movement).",
    "Levodopa converts to dopamine in brain, providing most effective Parkinson's symptom relief.",
    "Deep brain stimulation electrically modulates brain circuits, reducing Parkinson's motor symptoms.",
    "Multiple sclerosis involves autoimmune attack on nerve myelin, disrupting signal transmission.",
    "MS symptoms vary widely: weakness, numbness, vision problems, balance issues, cognitive changes.",
    "Disease-modifying therapies reduce MS relapses and slow disability progression.",
    "Epilepsy causes recurrent seizures from abnormal brain electrical activity.",
    "Antiepileptic drugs control 70% of epilepsy, working through various brain mechanisms.",
    "Status epilepticus is prolonged seizure requiring emergency treatment to prevent brain damage.",
    "Stroke symptoms require immediate emergency care to minimize permanent brain damage.",
    "TIA (transient ischemic attack) causes temporary stroke-like symptoms, warning of high stroke risk.",
    "Migraine causes severe headache with nausea and light sensitivity, often disabling.",
    "Triptans are effective migraine-specific medications, constricting blood vessels and blocking pain.",
    "Tension headache is most common headache type, caused by muscle tension and stress.",
    "Dementia is general term for cognitive decline severe enough to interfere with daily life.",
    "Vascular dementia results from reduced blood flow to brain, often following strokes.",
    "Peripheral neuropathy damages peripheral nerves, causing numbness, tingling, and pain in extremities.",
    "Carpal tunnel syndrome causes hand numbness from median nerve compression in wrist.",
    "Sciatica causes leg pain from sciatic nerve compression, often from herniated disc.",
    "Concussion is mild traumatic brain injury requiring rest and gradual return to activity.",
    "Meningitis inflames brain and spinal cord membranes, requiring urgent medical treatment.",
    "Brain imaging with CT or MRI helps diagnose stroke, tumors, and structural abnormalities.",
    "Cognitive behavioral therapy is effective treatment for many neurological and psychiatric conditions.",
    
    # CANCER (30 facts)
    "Cancer develops when cells acquire mutations enabling uncontrolled growth and spread.",
    "Breast cancer is most common cancer in women, with 1 in 8 lifetime risk.",
    "Mammography screening detects breast cancer early when treatment is most effective.",
    "BRCA1 and BRCA2 mutations greatly increase breast and ovarian cancer risk.",
    "Lung cancer is leading cancer killer, strongly associated with cigarette smoking.",
    "Lung cancer screening with low-dose CT reduces mortality in high-risk smokers.",
    "Colorectal cancer develops from polyps over years, making screening highly effective.",
    "Colonoscopy finds and removes precancerous polyps, preventing colorectal cancer development.",
    "Prostate cancer is most common cancer in men, typically slow-growing.",
    "PSA screening for prostate cancer is controversial, requiring informed individual decisions.",
    "Skin cancer is most common cancer type, with basal cell carcinoma most frequent.",
    "Melanoma is most dangerous skin cancer, potentially deadly but curable when caught early.",
    "ABCDE rule identifies melanoma: Asymmetry, Border, Color, Diameter, Evolution.",
    "Sun protection through sunscreen, protective clothing reduces skin cancer risk significantly.",
    "Leukemia is cancer of blood-forming tissues, causing abnormal white blood cell production.",
    "Lymphoma is cancer of lymphatic system, divided into Hodgkin and non-Hodgkin types.",
    "Hodgkin lymphoma is highly curable with chemotherapy and radiation, even in advanced stages.",
    "Pancreatic cancer has poor prognosis due to late detection and aggressive biology.",
    "Ovarian cancer often presents with vague symptoms, frequently diagnosed at advanced stage.",
    "Cervical cancer is preventable through HPV vaccination and regular Pap test screening.",
    "HPV causes most cervical cancers, with types 16 and 18 most oncogenic.",
    "Chemotherapy kills rapidly dividing cells, affecting cancer and causing side effects.",
    "Radiation therapy damages cancer cell DNA, causing death after accumulation of damage.",
    "Immunotherapy harnesses immune system to fight cancer, producing durable responses.",
    "Targeted therapy exploits specific cancer cell vulnerabilities with fewer side effects.",
    "Bone marrow transplant can cure some leukemias by replacing cancerous cells.",
    "Cancer staging guides treatment, ranging from localized disease to distant metastases.",
    "Palliative care improves quality of life for cancer patients through symptom management.",
    "Cancer survivors require long-term follow-up for recurrence detection and late effects.",
    "Clinical trials test new cancer treatments, offering access to cutting-edge therapies.",
    
    # INFECTIOUS DISEASE (30 facts)
    "Influenza virus causes seasonal respiratory illness, preventable with annual vaccination.",
    "Flu complications include pneumonia, myocarditis, and can be life-threatening.",
    "Antiviral medications like oseltamivir reduce flu duration if started within 48 hours.",
    "COVID-19 is caused by SARS-CoV-2 virus, spreading primarily through respiratory droplets.",
    "COVID-19 vaccines dramatically reduce infection severity, hospitalization, and death.",
    "Long COVID causes persistent symptoms including fatigue, brain fog, lasting months.",
    "HIV attacks immune system CD4 cells, progressively weakening body's defenses.",
    "AIDS is advanced HIV disease with CD4 count below 200 or opportunistic infections.",
    "Antiretroviral therapy suppresses HIV replication, preventing AIDS and transmission.",
    "Undetectable HIV viral load means virus cannot be transmitted sexually (U=U).",
    "Pre-exposure prophylaxis (PrEP) prevents HIV acquisition in high-risk individuals.",
    "Hepatitis B transmits through blood and body fluids, vaccine-preventable, can become chronic.",
    "Chronic hepatitis B can cause cirrhosis and liver cancer, treatable with antivirals.",
    "Hepatitis C transmits through blood, often chronic, now curable with direct-acting antivirals.",
    "Tuberculosis spreads through airborne droplets, requiring prolonged multi-drug treatment.",
    "Malaria is transmitted by mosquitoes, causing 400,000 deaths annually, mostly in Africa.",
    "Lyme disease is transmitted by deer ticks, causing rash and systemic symptoms.",
    "Sexually transmitted infections are rising, requiring screening and treatment to prevent complications.",
    "Chlamydia and gonorrhea are curable but often asymptomatic, causing infertility if untreated.",
    "Syphilis progresses through stages if untreated, potentially causing severe complications.",
    "HPV vaccine prevents genital warts and cancers, most effective before sexual debut.",
    "Urinary tract infections are common in women, causing dysuria, frequency, and urgency.",
    "Antibiotic resistance is growing threat, necessitating appropriate antibiotic use.",
    "C. difficile colitis causes severe diarrhea after antibiotics disrupt normal gut flora.",
    "MRSA is antibiotic-resistant staph causing difficult-to-treat skin and invasive infections.",
    "Sepsis is dysregulated immune response to infection causing organ dysfunction.",
    "Early sepsis recognition and treatment with antibiotics and fluids improves survival.",
    "Vaccination has eliminated or controlled many infectious diseases, saving millions of lives.",
    "Hand hygiene is single most important infection prevention measure.",
    "Respiratory etiquette including covering coughs reduces respiratory infection transmission.",
    
    # MENTAL HEALTH (30 facts)
    "Major depression affects 21 million US adults, causing persistent sadness and loss of interest.",
    "Depression symptoms include depressed mood, anhedonia, sleep changes, and guilt.",
    "SSRIs are first-line antidepressants with favorable side effect profile.",
    "Cognitive behavioral therapy changes negative thought patterns causing depression.",
    "Electroconvulsive therapy induces controlled seizure, highly effective for severe depression.",
    "Depression increases heart disease, stroke, and dementia risk through multiple pathways.",
    "Anxiety disorders affect 40 million Americans, causing excessive worry interfering with life.",
    "Generalized anxiety disorder involves persistent excessive worry about multiple life areas.",
    "Panic disorder causes recurrent unexpected panic attacks with intense fear.",
    "Social anxiety disorder involves intense fear of social situations and negative evaluation.",
    "Exposure therapy systematically confronts feared situations, highly effective for anxiety.",
    "Bipolar disorder causes extreme mood swings between manic highs and depressive lows.",
    "Mood stabilizers like lithium prevent bipolar mood episodes and reduce suicide risk.",
    "Schizophrenia affects thinking and perception, often involving delusions and hallucinations.",
    "Antipsychotic medications treat schizophrenia by modulating dopamine and other neurotransmitters.",
    "PTSD develops after trauma exposure, causing intrusive memories and hyperarousal.",
    "Trauma-focused therapies like prolonged exposure are first-line PTSD treatments.",
    "OCD involves unwanted intrusive thoughts (obsessions) and repetitive behaviors (compulsions).",
    "Exposure and response prevention is gold-standard OCD treatment.",
    "ADHD affects 8% of children and 4% of adults, characterized by inattention and impulsivity.",
    "Stimulant medications like methylphenidate are first-line ADHD treatment.",
    "Eating disorders include anorexia, bulimia, and binge eating disorder.",
    "Anorexia nervosa involves restrictive eating and distorted body image.",
    "Eating disorders have highest mortality of psychiatric conditions.",
    "Substance use disorders involve compulsive drug use despite harmful consequences.",
    "Medication-assisted treatment combines therapy with medications for addiction.",
    "Mental health stigma prevents many from seeking help though conditions are treatable.",
    "Suicide is preventable, with warning signs including hopelessness and withdrawal.",
    "Crisis resources like 988 Suicide Lifeline provide immediate mental health support.",
    "Psychotherapy treats mental health conditions through evidence-based talk therapy approaches.",
    
    # GI, RENAL, MUSCULOSKELETAL (30 facts)
    "GERD occurs when stomach acid flows back into esophagus, irritating its lining.",
    "Proton pump inhibitors are most effective GERD medications, healing esophagitis.",
    "Inflammatory bowel disease includes Crohn's disease and ulcerative colitis.",
    "Celiac disease is autoimmune reaction to gluten, requiring strict gluten-free diet.",
    "Cirrhosis is end-stage liver scarring from chronic damage, leading to liver failure.",
    "Non-alcoholic fatty liver disease affects 25% of population, linked to obesity.",
    "Chronic kidney disease is progressive kidney function loss over months to years.",
    "Diabetes and hypertension cause most chronic kidney disease cases.",
    "Dialysis removes waste and excess fluid when kidneys fail.",
    "Kidney transplant is preferred treatment for kidney failure, offering better outcomes than dialysis.",
    "Acute kidney injury is sudden kidney failure, often reversible if treated promptly.",
    "Urinary tract infections cause dysuria and frequency, treated with antibiotics.",
    "Kidney stones form from mineral crystallization, causing severe pain when passing.",
    "Osteoarthritis is cartilage breakdown from wear and tear, causing joint pain and stiffness.",
    "Rheumatoid arthritis is autoimmune joint inflammation, treated with disease-modifying drugs.",
    "Osteoporosis weakens bones through decreased density, dramatically increasing fracture risk.",
    "DEXA scan measures bone density, diagnosing osteoporosis and assessing fracture risk.",
    "Bisphosphonates are first-line osteoporosis drugs, reducing fracture risk.",
    "Gout causes sudden severe joint pain from uric acid crystal deposition.",
    "Back pain affects 80% of people during lifetime, usually self-limiting.",
    "Herniated disc causes pain when disc material compresses nerve root.",
    "Physical therapy helps restore movement and function after musculoskeletal injury.",
    "Hip replacement surgery treats severe arthritis and joint damage.",
    "Knee replacement surgery provides pain relief and improved function for severe arthritis.",
    "Fibromyalgia causes widespread musculoskeletal pain, fatigue, and sleep problems.",
    "Carpal tunnel syndrome causes hand numbness from median nerve compression.",
    "Rotator cuff tears cause shoulder pain and weakness, common in older adults.",
    "Stress fractures result from repetitive force, common in runners and athletes.",
    "Tendinitis is inflammation of tendon, causing pain and limited movement.",
    "Plantar fasciitis causes heel pain from inflammation of foot arch ligament.",
    
    # WOMEN'S HEALTH, PEDIATRICS, GENERAL (40 facts)
    "Pregnancy involves 40 weeks of fetal development divided into three trimesters.",
    "Prenatal care includes regular checkups monitoring mother and baby's health.",
    "Folic acid supplementation before conception prevents neural tube defects.",
    "Gestational diabetes screening occurs at 24-28 weeks, requiring glucose control.",
    "Preeclampsia causes hypertension and organ dysfunction after 20 weeks.",
    "Breastfeeding provides optimal infant nutrition and maternal health benefits.",
    "Postpartum depression affects 1 in 7 women, requiring recognition and treatment.",
    "Contraception options include hormonal methods, IUDs, barrier methods, and sterilization.",
    "Menopause transition brings hormonal changes causing hot flashes and bone loss.",
    "Cervical cancer screening with Pap tests prevents cancer through early detection.",
    "Childhood vaccinations protect against 14 serious diseases following CDC schedule.",
    "Vaccine safety is extensively studied, with benefits far outweighing minimal risks.",
    "Well-child visits monitor growth, development, and provide anticipatory guidance.",
    "Sudden infant death syndrome risk reduced by back sleeping and safe environment.",
    "Autism spectrum disorder varies widely, requiring individualized interventions.",
    "ADHD in children managed with medication, behavioral therapy, and school support.",
    "Childhood obesity prevention includes healthy eating, activity, and limiting screens.",
    "Developmental screening identifies delays enabling early intervention.",
    "Regular exercise reduces risk of chronic diseases and improves mental health.",
    "Balanced diet supports overall health and disease prevention.",
    "Smoking cessation dramatically reduces risk of cancer, heart disease, and stroke.",
    "Adequate sleep 7-9 hours nightly is essential for physical and mental health.",
    "Stress management through meditation and exercise improves health outcomes.",
    "Preventive care includes regular health screenings and vaccinations.",
    "Blood pressure should be checked at least every 2 years starting at age 18.",
    "Cholesterol screening recommended starting at age 35 for men, 45 for women.",
    "Colorectal cancer screening should begin at age 45 for average-risk individuals.",
    "Mammography screening for breast cancer typically begins at age 40-50.",
    "Bone density screening for osteoporosis recommended for women at age 65.",
    "Annual flu vaccination recommended for everyone 6 months and older.",
    "Healthy lifestyle habits prevent up to 80% of premature heart disease and stroke.",
    "Regular primary care visits enable early disease detection and management.",
    "Health insurance coverage improves access to preventive and treatment services.",
    "Advance care planning documents healthcare preferences for future decision-making.",
    "Living will specifies end-of-life medical treatment preferences.",
    "Healthcare proxy designates someone to make medical decisions if unable.",
    "Telemedicine provides healthcare remotely using technology, improving access.",
    "Electronic health records store patient information digitally, improving care coordination.",
    "Patient privacy is protected by HIPAA regulations in healthcare settings.",
    "Informed consent ensures patients understand treatment risks and benefits before proceeding.",
]


def main():
    print("\n" + "="*70)
    print("LOCAL PERMANENT LAYER - 300 ESSENTIAL FACTS")
    print("="*70 + "\n")
    
    print("📊 Configuration (Optimized for 5K cloud):")
    print(f"   Facts: {len(PERMANENT_FACTS_300)}")
    print(f"   Capacity: 30% of local VDB (1,000 total)")
    print(f"   Storage: Disk (permanent, never evicted)")
    print(f"   RAM usage: Loaded on-demand (~525 KB)")
    print(f"   Purpose: Offline baseline + core medical knowledge\n")
    
    # Create directory
    Path(BASE_LAYER_PATH).mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("[1/3] Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model ready\n")
    
    # Generate embeddings
    print("[2/3] Generating embeddings for 300 facts...")
    embeddings = model.encode(
        PERMANENT_FACTS_300, 
        show_progress_bar=True, 
        normalize_embeddings=True
    )
    print(f"✅ Generated {len(embeddings)} embeddings\n")
    
    # Create FAISS index
    print("[3/3] Creating FAISS index...")
    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    index.add(embeddings.astype('float32'))
    
    # Save index
    index_path = Path(BASE_LAYER_PATH) / "partition_0.index"
    faiss.write_index(index, str(index_path))
    print(f"✅ Saved index: {index_path}")
    
    # Save metadata
    metadata = {
        str(i): {
            "partition_idx": 0,
            "local_idx": i,
            "text": PERMANENT_FACTS_300[i]
        }
        for i in range(len(PERMANENT_FACTS_300))
    }
    
    metadata_path = Path(BASE_LAYER_PATH) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata: {metadata_path}\n")
    
    print("="*70)
    print("✅ LOCAL PERMANENT LAYER READY")
    print("="*70)
    print(f"\n💾 Location: {BASE_LAYER_PATH}")
    print(f"📊 Vectors: 300 essential medical facts")
    print(f"💿 Disk size: ~525 KB")
    print(f"🔒 Type: Permanent (never evicted)")
    print(f"📈 Capacity: 30% of 1K local VDB")
    print(f"🎯 Dynamic space remaining: 700 vectors for learning\n")
    print("Ready for hybrid VDB system! 🚀\n")


if __name__ == "__main__":
    main()