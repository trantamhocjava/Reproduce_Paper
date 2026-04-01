explicid_isic_dict = {
    "color": [
        "highly variable, often with multiple colors (black, brown, red, white, blue)",
        "uniformly tan, brown, or black",
        "translucent, pearly white, sometimes with blue, brown, or black areas",
        "red, pink, or brown, often with a scale",
        "light brown to black",
        "pink brown or red",
        "red, purple, or blue",
    ],
    "shape": ["irregular", "round", "round to irregular", "variable"],
    "border": [
        "often blurry and irregular",
        "sharp and well-defined",
        "rolled edges, often indistinct",
    ],
    "dermoscopic patterns": [
        "atypical pigment network, irregular streaks, blue-whitish veil, irregular",
        "regular pigment network, symmetric dots and globules",
        "arborizing vessels, leaf-like areas, blue-gray avoid nests",
        "strawberry pattern, glomerular vessels, scale",
        "cerebriform pattern, milia-like cysts, comedo-like openings",
        "central white patch, peripheral pigment network",
        "depends on type (e.g., cherry angiomas have red lacunae; spider angiomas have a central red dot with radiating legs",
    ],
    "texture": [
        "a raised or ulcerated surface",
        "smooth",
        "smooth, possibly with telangiectasias",
        "rough, scaly",
        "warty or greasy surface",
        "firm, may dimple when pinched",
    ],
    "symmetry": [
        "asymmetrical",
        "symmetrical",
        "can be symmetrical or asymmetrical depending on type",
    ],
    "elevation": [
        "flat to raised",
        "raised with possible central ulceration",
        "slightly raised",
        "slightly raised maybe thick",
    ],
}

explicid_idrid_dict = {
    "microaneurysms": [
        "absent",
        "few isolated microaneurysms",
        "multiple microaneurysms in several regions",
        "numerous microaneurysms in all quadrants",
    ],
    "hemorrhages": [
        "absent",
        "few dot hemorrhages",
        "multiple dot-blot hemorrhages",
        "extensive hemorrhages in multiple quadrants",
        "pre-retinal or vitreous hemorrhage",
    ],
    "exudates": [
        "absent",
        "few hard exudates",
        "multiple hard exudates near vessels or macula",
        "extensive hard exudates",
    ],
    "cotton_wool_spots": [
        "absent",
        "few localized cotton-wool spots",
        "multiple cotton-wool spots indicating ischemia",
    ],
    "vascular_abnormalities": [
        "normal retinal vessels",
        "mild vessel dilation or tortuosity",
        "venous beading in limited areas",
        "venous beading in multiple quadrants",
        "intraretinal microvascular abnormalities (IRMA)",
    ],
    "neovascularization": [
        "absent",
        "suspected abnormal fine vessels",
        "neovascularization elsewhere (NVE)",
        "neovascularization at the disc (NVD)",
    ],
    "distribution_of_lesions": [
        "no lesions present",
        "localized to one region",
        "present in multiple regions",
        "present in all four quadrants",
    ],
    "overall_severity_pattern": [
        "normal retina",
        "early diabetic changes",
        "moderate non-proliferative diabetic retinopathy",
        "severe non-proliferative diabetic retinopathy",
        "proliferative diabetic retinopathy",
    ],
}


explicid_busi_dict = {
    "shape": [
        "round or oval",
        "irregular",
    ],
    "margin": [
        "smooth, well-circumscribed",
        "microlobulated",
        "indistinct or spiculated",
    ],
    "orientation": [
        "parallel to skin (wider than tall)",
        "non-parallel (taller than wide)",
    ],
    "echo_pattern": [
        "anechoic or hypoechoic homogeneous",
        "heterogeneous hypoechoic",
        "complex echogenicity",
    ],
    "posterior_acoustic_features": [
        "no posterior feature",
        "posterior enhancement",
        "posterior shadowing",
    ],
    "boundary": [
        "well-defined boundary",
        "ill-defined boundary",
    ],
    "calcification": [
        "absent",
        "present",
    ],
}


explicid_nct_dict = {
    "overall_stain_balance": [
        "very pale / low stain signal (near-background)",
        "eosin-dominant pink (fibers / muscle / collagen)",
        "hematoxylin-dominant purple-blue (nuclei-rich)",
        "mixed pink-purple with clear lumens/structures (glandular)",
        "heterogeneous dirty mix with debris/necrosis-like tones",
    ],
    "tissue_occupancy": [
        "almost no tissue (mostly blank background)",
        "sparse tissue fragments / edge tissue",
        "moderate tissue coverage",
        "high tissue coverage (field filled with tissue)",
    ],
    "dominant_architecture": [
        "honeycomb of large clear vacuoles (adipose-like)",
        "parallel or fascicular linear bundles (muscle-like)",
        "fibrous web / collagenous matrix (stroma-like)",
        "round/oval glandular lumens (crypt/gland-like)",
        "amorphous clumps without clear architecture (debris-like)",
        "sheet/nodules of small nuclei (lymphoid-like)",
        "large pale pools/lakes in spaces (mucin-like)",
    ],
    "nuclear_density": [
        "very low (few nuclei visible)",
        "low",
        "moderate",
        "high",
        "very high (packed small nuclei)",
    ],
    "lumen_or_space_character": [
        "none or minimal open spaces",
        "many large round clear spaces with thin walls (fat vacuoles)",
        "gland lumens: round/oval and regular",
        "gland lumens: irregular, fused, angulated, crowded",
        "pale mucin pools/lakes (smooth, translucent)",
        "irregular holes with dirty content (necrotic/debris lumens)",
    ],
    "cell_shape_and_distribution": [
        "few cells; scattered nuclei only",
        "spindle-shaped nuclei dispersed in matrix",
        "elongated nuclei aligned with fibers (muscle)",
        "small round uniform nuclei in sheets/aggregates (lymphoid)",
        "epithelial cells lining lumens (glands)",
        "pleomorphic crowded epithelial nuclei (tumor-like)",
        "fragmented nuclear dust / smudgy material (debris)",
    ],
    "surface_texture_or_coarseness": [
        "smooth / low texture (background or mucin pools)",
        "fine dotted texture (many small nuclei)",
        "fibrous / streaky texture (collagen or muscle)",
        "coarse granular / dirty texture (debris/necrosis)",
        "complex heterogeneous glandular texture (tumor-like)",
    ],
}


explicid_lungcolon_dict = {
    # 0
    "organ_context": [
        "colonic mucosa with crypt-based architecture / gland-rich mucosa",
        "lung parenchyma with alveoli/bronchiolar structures",
        "organ context not clear / mixed field",
    ],
    # 1
    "overall_architecture": [
        "infiltrative irregular glands / back-to-back glands (invasion pattern)",
        "gland-forming adenocarcinoma pattern in lung (acinar/papillary/lepidic-like)",
        "solid nests/sheets of tumor cells (non-glandular dominant)",
        "orderly benign architecture (preserved normal tissue pattern)",
    ],
    # 2
    "cytologic_atypia": [
        "minimal atypia (near-normal nuclei, uniform cells)",
        "moderate atypia (noticeable pleomorphism, some hyperchromasia)",
        "marked atypia (high pleomorphism, prominent nucleoli, hyperchromasia)",
    ],
    # 3
    "mitotic_activity": [
        "low/rare mitoses",
        "intermediate mitoses",
        "high/frequent mitoses",
    ],
    # 4
    "necrosis": [
        "absent",
        "focal necrosis",
        "extensive necrosis",
    ],
    # 5
    "mucin_features": [
        "no evident mucin",
        "intracellular or luminal mucin (e.g., goblet cells / luminal secretion)",
        "abundant extracellular mucin pools",
    ],
    # 6
    "squamous_differentiation": [
        "absent (no keratinization / no intercellular bridges)",
        "present (keratinization and/or intercellular bridges; possible keratin pearls)",
        "benign squamous metaplasia / superficial squamous change (non-malignant)",
    ],
}
