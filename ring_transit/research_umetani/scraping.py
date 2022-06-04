from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import csv
import time

list = os.listdir()

#obsids = [i for i in list if i.isdecimal()]
driver = webdriver.Chrome("/Users/u_tsubasa/work/chromedriver")

csv_list = []
# グーグルを開く
driver.get("https://www.kaigokensaku.mhlw.go.jp/13/index.php?action_kouhyou_pref_search_bookmark_list=true")
import pdb;pdb.set_trace()
#submit = driver.find_element_by_id("SearchFormPanel-submitButton")

#submit.click()
start_time = driver.find_elements_by_xpath('/html/body/form/div/table/tbody/tr/td[18]')[0].text
duration = driver.find_elements_by_xpath('/html/body/form/div/table/tbody/tr/td[8]')[0].text
instrument = driver.find_elements_by_xpath('/html/body/form/div/table/tbody/tr/td[5]')[0].text
ra = driver.find_elements_by_xpath('/html/body/form/div/table/tbody/tr/td[11]')[0].text
dec = driver.find_elements_by_xpath('/html/body/form/div/table/tbody/tr/td[12]')[0].text
target_name = driver.find_elements_by_xpath('/html/body/form/div/table/tbody/tr/td[9]')[0].text
info_list = [target_name, obsid, instrument, ra, dec, start_time, duration]
csv_list.append(info_list)
#import pdb; pdb.set_trace()
print(obsid)
print(info_list)



with open('/Users/u_tsubasa/work/chandra/archive/chandra_obs_info2.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Target name', 'ObsID', 'Instrument', 'RA', 'DEC', 'Start time (UT)', 'Duration (ks)'])
    writer.writerows(csv_list)


"""
star_names = ['HD 102117','HD 107148','HD 109749','HD 134987','HD 141004','HD 168746','HD 178911B','HD 179949','HD 185269','HD 187123','HD 188015','HD 204313','HD 20794','HD 209458','HD 28185','HD 30177','HD 39091','HD 41004','HD 46375','HD 49674','HD 6434','HD 97658','YZ Cet']
driver = webdriver.Chrome("/Users/u_tsubasa/work/chandra/archive/driver/chromedriver")

csv_list = []

for i, star_name in enumerate(star_names):
    if i > 0:
        # 新しいタブ
        driver.execute_script("window.open('','_blank');")
        driver.switch_to.window(driver.window_handles[i])
    # グーグルを開く
    driver.get("https://gea.esac.esa.int/archive/")
    WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "gaia-MainLayoutPanel-ContentLayoutPanel-HomePanel-Anchor-searchAnchor")))

    search = driver.find_element_by_id('gaia-MainLayoutPanel-ContentLayoutPanel-HomePanel-Anchor-searchAnchor')
    search.click()

    # 検索ワード入力
    search_box = driver.find_element_by_id("gaia-MainLayoutPanel-ContentLayoutPanel-CataloguePanel-SearchFormPanel-GeometricalSearchPanel-PositionOptionsPanel-DefaultTextBox-nameTextBox")
    search_words = star_name
    #import pdb; pdb.set_trace()
    search_box.send_keys(search_words)

    # 検索実行
    time.sleep(3)
    judge = driver.find_element_by_xpath('/html/body/div[2]/div[2]/div/div[4]/div/div[3]/div/div[3]/div/div[2]/div/div/table/tbody/tr[1]/td/table/tbody/tr[2]/td/div/div[1]/div/div[3]/div/div[3]/table/tbody/tr/td[1]/div/div[1]/table/tbody/tr[2]/td/table/tbody/tr/td/div')
    if 'Not' in judge.text:
        print(f'{star_name} is no data')
        continue
    else:
        pass
    submit = driver.find_element_by_id("SearchFormPanel-submitButton")

    submit.click()
    WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "gaia-2-TableView-CellTable-table")))
    time.sleep(2)
    parallax = driver.find_element_by_xpath('/html/body/div[2]/div[2]/div/div[4]/div/div[3]/div/div[3]/div/div[4]/div/div[2]/div/div[2]/div/div[4]/div/div[2]/div/div[2]/div/div[2]/div/div[2]/div/div[4]/div/div/table/tbody[1]/tr/td[4]/div').text
    parallax_err = driver.find_element_by_xpath('/html/body/div[2]/div[2]/div/div[4]/div/div[3]/div/div[3]/div/div[4]/div/div[2]/div/div[2]/div/div[4]/div/div[2]/div/div[2]/div/div[2]/div/div[2]/div/div[4]/div/div/table/tbody[1]/tr/td[5]/div').text
    info_list = [star_name, parallax, parallax_err]
    csv_list.append(info_list)
    #import pdb; pdb.set_trace()
    print(star_name)
    print(info_list)



with open('/Users/u_tsubasa/work/chandra/archive/gaia_obs_info.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Target name', 'parallax', 'parallax_err'])
    writer.writerows(csv_list)
"""
