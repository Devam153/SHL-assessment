import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


def scrape_descriptions(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Scrape the detailed description text from each SHL assessment page and
    add it as a new column to the catalog CSV.

    Args:
        csv_path: Path to input CSV containing at least a 'Link' column.
        output_path: Path to save enriched CSV. If None, appends '_with_desc.csv'.

    Returns:
        DataFrame with an added 'Description' column.
    """
    # Load the catalog
    df = pd.read_csv(csv_path)

    # Prepare Description column
    if 'Description' not in df.columns:
        df['Description'] = None

    # Iterate through each URL and scrape
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Scraping descriptions'):
        url = row.get('Link') or row.get('link')
        if not url:
            continue

        # Skip if already has description
        if pd.notna(row['Description']) and str(row['Description']).strip():
            continue

        try:
            resp = requests.get(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; SHL-Desc-Scraper/1.0)'},
                timeout=10
            )
            resp.encoding = 'utf-8'
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Strategy: find the 'Description' <h4> section first
            desc_text = ''
            # 1. Look for an <h4> with text 'Description'
            h4 = soup.find('h4', string=re.compile(r'Description', re.I))
            if h4:
                # get the very next <p> sibling
                p_desc = h4.find_next_sibling('p')
                if p_desc:
                    desc_text = p_desc.get_text(' ', strip=True)

            # 2. Fallback: find the main heading <h1>, then collect subsequent <p> tags
            if not desc_text:
                h1 = soup.find('h1')
                if h1:
                    for sibling in h1.find_next_siblings():
                        if sibling.name != 'p':
                            break
                        desc_text += sibling.get_text(strip=True) + ' '
                    desc_text = desc_text.strip()

            # 3. Another fallback: look for a div with 'description' class
            if not desc_text:
                div = soup.find('div', class_=re.compile(r'description', re.I))
                if div:
                    desc_text = div.get_text(' ', strip=True)

            # Assign cleaned description
            df.at[idx, 'Description'] = desc_text or None

        except Exception as e:
            print(f"[!] Error scraping row {idx} ({url}): {e}")

        # Politeness delay
        time.sleep(0.5)

    # Determine output path
    if not output_path:
        if csv_path.lower().endswith('.csv'):
            output_path = csv_path[:-4] + '_with_desc.csv'
        else:
            output_path = csv_path + '_with_desc.csv'

    # Save enriched CSV
    df.to_csv(output_path, index=False)
    print(f"Descriptions saved to {output_path}")

    return df


if __name__ == '__main__':
    # Example usage - adjust paths as needed
    scrape_descriptions(
        csv_path='src/data/shl_full_catalog_with_duration.csv',
        output_path='src/data/shl_full_catalog_with_duration&desc.csv'
    )