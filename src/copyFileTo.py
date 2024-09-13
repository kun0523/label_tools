import os
import re
import shutil

def createDir(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        print(f"Target_Dir: {target_dir} already exists!!")
    return target_dir


if __name__ == "__main__":

    src_dir = r"E:\DataSets\edge_crack\iqc_crack"
    # src_dir = r"E:\DataSets\edge_crack\0823\原图0822"
    dst_dir = r"E:\DataSets\edge_crack\tmp"

    for parent_dir, sub_dir, file_lst in os.walk(src_dir):
        # print("parent_dir: ", parent_dir)
        part = re.search(r"([UD]\d)", parent_dir)
        date_match = re.search(r"(202\d.\d+.\d+)", parent_dir)
        check_res = re.search(r"([ON][KG])", parent_dir)
        for file in file_lst:
            if(not file.endswith(".jpg")): continue
            if(part):
                print(f"------------- part:{part.group(1)} ----------")
                part_str = part.group(1)
            else:
                part_str = "unk_part"

            if(date_match):
                print(f"------------- date:{date_match.group(1)} ----------")
                date_str = date_match.group(1)
            else:
                part_str = "unk_date"

            if(check_res):
                print(f"------------- check_res:{check_res.group(1)} ----------")
                check_res_str = check_res.group(1)
            else:
                check_res_str = "unk"

            file_id = re.search(r"^(\d+)", file).group(1)
            print(f"==== file_id:{file_id}")
            src_path = os.path.join(parent_dir, file)
            print("src_file_name: ", src_path)

            tmp_save_dir = os.path.join(dst_dir, date_str.replace(".", "_"))
            createDir(tmp_save_dir)

            save_name = f"{file_id}_{part_str}_{check_res_str}.jpg"
            dst_path = os.path.join(tmp_save_dir, save_name)
            print("dst_file_path: ", dst_path)
            shutil.copyfile(src_path, dst_path)


    print("done")