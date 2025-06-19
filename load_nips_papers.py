import pandas as pd
import requests
import time

from argparse import ArgumentParser, SUPPRESS
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin

# Set up command line execution
parser = ArgumentParser(
    add_help=False, description="Script for parsing NeurlIPS papers"
)

required = parser.add_argument_group("required arguments")
optional = parser.add_argument_group("optional arguments")

optional.add_argument(
    "-h",
    "--help",
    action="help",
    default=SUPPRESS,
    help="show this help message and exit",
)

required.add_argument(
    "-d", "--dir-paper", required=True, type=str, help="paper save directory"
)
required.add_argument("-y", "--year", required=True, type=str, help="conference year")

optional.add_argument(
    "-m", "--meta-df-path", type=str, help="save path for the paper metadata DataFrame in .parquet format"
)
optional.add_argument("-p", "--proxy-path", type=str, help="proxy csv file path")
optional.add_argument(
    "--no-meta-df", action="store_false", help="do not save paper metadata dataframe"
)
optional.add_argument("--no-proxy", action="store_false", help="do not use proxy")

args = parser.parse_args()

if args.no_meta_df and not args.meta_df_path:
    parser.error("--meta-df-path can only be skipped when --no-meta-df is used.")
if args.no_proxy and not args.proxy_path:
    parser.error("--proxy-path can only be skipped when --no-proxy is used.")

# Constants
headers = {
    "Accept": "text/html,application/pdf",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
}


# Filter out non-working proxies
def test_proxy(proxy):
    try:
        response = requests.get(
            "https://httpbin.org/ip",
            headers=headers,
            proxies={"http": proxy, "https": proxy},
            timeout=5,
        )
        if response.status_code == 200:
            return True
    except:
        pass
    return False


# Each NeurlIPS paper has a separate page with metadata and pdf link
# Create a list of paper page urls
def load_nips_paper_page_urls(year, header):
    print(f"Loading year {year} paper page urls...")
    url = "https://papers.nips.cc/paper_files/paper/"
    base_url = "https://papers.nips.cc"
    resp = requests.get(url + year, headers=header)
    soup = BeautifulSoup(resp.text, "html.parser")
    paper_urls = [
        urljoin(base_url, e["href"])
        for e in soup.find_all("a", href=True)
        if "/paper/" + year + "/" in e["href"]
    ]
    return paper_urls


# Recursive function trying proxies
def try_proxy(url, i, proxy_list, header):
    if i >= len(proxy_list):
        if len(proxy_list) > 0:
            try:
                resp = requests.get(url, headers=header, timeout=(7, 7))
                i = 0
                return i, resp
            except:
                i = 0
                return i, "error"
        else:
            i = 0
            return i, "error"
    else:
        proxies = {
            "http": proxy_list[i],
            "https": proxy_list[i],
        }
        try:
            resp = requests.get(url, headers=header, proxies=proxies, timeout=(7, 7))
            return i, resp
        except:
            i += 1
            return try_proxy(url, i, proxy_list, header)


# Create a dataframe of paper metadata
def load_nips_paper_meta(paper_urls, proxy_list, year, header, save_df, name_df):
    print(f"Loading year {year} paper metadata...")
    base_url = "https://papers.nips.cc"
    title_list = []
    abstract_list = []
    author_list = []
    hash_list = []
    pdf_list = []
    error_list = []
    i = 0
    for paper_url in tqdm(paper_urls):
        try:
            resp = requests.get(paper_url, headers=header, timeout=(7, 7))
        except:
            i, resp = try_proxy(paper_url, i, proxy_list, header)
        if resp == "error":
            error_list.append(paper_url)
        else:
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            abstract_list.append(
                soup.find("h4", string="Abstract").find_next_sibling("p").get_text()
            )
            author_list.append(
                soup.find("h4", string="Authors").find_next_sibling("p").get_text()
            )
            pdf_tag = soup.find("a", string="Paper")
            if pdf_tag and pdf_tag["href"].endswith(".pdf"):
                pdf_url = urljoin(base_url, pdf_tag["href"])
                pdf_list.append(pdf_url)
            else:
                pdf_list.append("NO DATA")
            hash_list.append(
                pdf_url[paper_url.rfind("/") + 1 : paper_url.find("-")]
                + "_"
                + str(year)
            )
            title_list.append(soup.title.string)
            time.sleep(0.5)
    df = pd.DataFrame(
        {
            "title": title_list,
            "abstract": abstract_list,
            "author": author_list,
            "year": year,
            "hash": hash_list,
            "pdf_url": pdf_list,
        }
    )
    if args.no_meta_df:
        df.to_parquet(args.meta_df_path)
    if error_list:
        print("Error list:")
        for j in error_list:
            print(j)
    return df


# Load paper pdfs
def load_nips_paper_pdfs(paper_urls, proxy_list, year, header, directory):
    print(f"Loading year {year} papers...")
    error_list = []
    i = 0
    for url in tqdm(paper_urls):
        try:
            resp = requests.get(url, headers=header, timeout=(7, 7))
        except:
            i, resp = try_proxy(url, i, proxy_list, header)
        if resp == "error":
            error_list.append(url)
        else:
            file_name = url[url.rfind("/") + 1 : url.find("-")] + "_" + str(year)
            with open(f"{directory}/{file_name}.pdf", "wb") as f:
                f.write(resp.content)
    if error_list:
        print("Error list:")
        for j in error_list:
            print(j)


# Run script
if args.no_proxy:
    print(f"Checking proxy list...")
    proxy_df = pd.read_csv(args.proxy_path, header=None).rename(columns={0: "proxy"})
    proxy_list = list(proxy_df.proxy)
    proxy_list = [p for p in tqdm(proxy_list) if test_proxy(p)]
    print(f"{len(proxy_list)}/{proxy_df.shape[0]} working proxies")
    for i in proxy_list:
        print("Working proxy:", i)
else:
    proxy_list = []

paper_urls = load_nips_paper_page_urls(args.year, headers)

paper_df = load_nips_paper_meta(
    paper_urls, proxy_list, args.year, headers, args.no_meta_df, args.meta_df_path
)

load_nips_paper_pdfs(
    paper_df.loc[:, "pdf_url"], proxy_list, args.year, headers, args.dir_paper
)
