from kltn_utils.cbm import const as cbm_const

if __name__ == "__main__":
    class_concept = cbm_const.CLASS_AND_CONCEPT["lcc"]

    print(f"concepts: {class_concept['concepts']} \n")
    print(f"concept2class: {class_concept['concept2class']}\n ")
    print(f"class_names: {class_concept['class_names']} \n")

    print("DONE")


ecbmplcode.train_ecbm
