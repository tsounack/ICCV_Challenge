import csv

diseases = ["Atelectasis", "Calcification of the Aorta", "Cardiomegaly", "Consolidation", "Edema", "Emphysema",
            "Enlarged Cardiomediastinum", "Fibrosis", "Fracture", "Hernia", "Infiltration", "Lung Lesion",
            "Lung Opacity", "Mass", "No Finding", "Nodule", "Pleural Effusion", "Pleural Other", "Pleural Thickening",
            "Pneumomediastinum", "Pneumonia", "Pneumoperitoneum", "Pneumothorax", "Subcutaneous Emphysema",
            "Support Devices", "Tortuous Aorta"]

for mode in ["train", "development"]:
    # Define the input and output file paths
    input_file = mode + '.csv'

    out = mode if mode == "train" else "test"
    output_path_file = out + '.image.tok'
    output_disease_file = out + '.label.tok'

    # Open the input file and read the CSV data
    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        # Open the output files for writing
        with open(output_path_file, 'w') as path_file, open(output_disease_file, 'w') as disease_file:
            # Write the path and disease columns to their respective files
            for row in reader:

                label = [disease for disease in diseases if disease in row and int(row[disease]) == 1]
                if not label:
                    label = ["No positive label"]

                disease_file.write(','.join(label) + '\n')
                path_file.write(row['path'] + '\n')


im = [l.strip() for l in open("train.image.tok").readlines()]
lab = [l.strip() for l in open("train.label.tok").readlines()]

im_test = [l.strip() for l in open("test.image.tok").readlines()]
lab_test = [l.strip() for l in open("test.label.tok").readlines()]

sixty = int(0.6 * len(im))
twenty = int(0.2 * len(im))
eighty = sixty + twenty

open("train.image.tok", "w").write("\n".join(im[:eighty]))
open("validate.image.tok", "w").write("\n".join(im[eighty:]))
open("test.image.tok", "w").write("\n".join(im_test))

open("train.label.tok", "w").write("\n".join(lab[:eighty]))
open("validate.label.tok", "w").write("\n".join(lab[eighty:]))
open("test.label.tok", "w").write("\n".join(lab_test))
