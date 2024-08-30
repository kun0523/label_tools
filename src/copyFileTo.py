import os
import re
import shutil



if __name__ == "__main__":

    src_dir = r"E:\DataSets\edge_crack\original_iqc\img_0827"
    # src_dir = r"E:\DataSets\edge_crack\0823\原图0822"
    dst_dir = r"E:\DataSets\edge_crack\original_iqc\tmp"

    for parent_dir, sub_dir, file_lst in os.walk(src_dir):
        # print("parent_dir: ", parent_dir)
        part = re.search(r"([UD]\d)", parent_dir)
        check_res = re.search(r"([ON][KG])", parent_dir)
        for file in file_lst:
            if(part):
                print(f"------------- part:{part.group(1)} ----------")
                part_str = part.group(1)
            else:
                part_str = "unknow_part"
            if(check_res):
                print(f"------------- check_res:{check_res.group(1)} ----------")
                check_res_str = check_res.group(1)
            else:
                check_res_str = "unk"

            file_id = re.search(r"^(\d{13})", file).group(1)
            print(f"==== file_id:{file_id}")
            src_path = os.path.join(parent_dir, file)
            print("src_file_name: ", src_path)
            save_name = f"{file_id}_{part_str}_{check_res_str}.jpg"
            dst_path = os.path.join(dst_dir, save_name)
            print("dst_file_path: ", dst_path)
            shutil.copyfile(src_path, dst_path)


    print("done")