import glob
import shutil

# 1つ上の階層のディレクトリパスを取得
dirs = glob.glob(
    "/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/lmfit_res/sap_0605_p0.05/transit_fit/*"
)

# 1つ上の階層のディレクトリをループ
for directory in dirs:
    # ディレクトリ内のファイルを取得
    files = glob.glob(directory + "/*")

    # ファイル名を"_"で分割し、最後の要素を抽出してソートする
    sorted_files = sorted(
        files, key=lambda x: int(x.split("/")[-1].split("_")[1]), reverse=True
    )

    # ソートされたファイルの1番目を
    cp_file = sorted_files[0]
    # "/mwork2/umetanitb/research_umetani/lmfit_res/sap_0605_p0.05/best_transit_fit"にコピー
    shutil.copyfile(
        cp_file,
        f"/mwork2/umetanitb/research_umetani/lmfit_res/sap_0605_p0.05/best_transit_fit/{cp_file.split('/')[-1]}",
    )
