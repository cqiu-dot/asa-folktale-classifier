#!/usr/bin/env python
"""
Data collection helper: Scripts to download and organize folktale data
"""

import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_data_directories():
    """Create all required data directories"""
    regions = [
        'china',    # Matches china_china_fables_dataset.json
        'korea',    # Matches korea_korea_fables_dataset.json
        'japan',    # Matches japan_japan_fables_dataset.json
        'vietnamese',
        'tibetan',
        'mongolian',
        'thai',
        'khmer',
        'malay',
        'filipino',
        'burmese',
        'lao',
    ]
    
    data_dir = Path('../')  # Parent directory (Downloads folder)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✓ Data directory: {data_dir.absolute()}")
    logger.info("✓ Place JSON files directly in the Downloads folder")
    logger.info("  Expected format: <region>_<region>_fables_dataset.json")
    logger.info("  Examples: china_china_fables_dataset.json, korea_korea_fables_dataset.json")


def download_western_data():
    """
    Download Western folktales dataset from GitHub
    
    Instructions to manually download:
    1. Visit: https://github.com/j-hagedorn/trilogy
    2. Navigate to: data/aft.csv
    3. Click "Raw" and save the file
    4. Place in: data/western/aft.csv
    """
    logger.info("To download Western folktales (AFT dataset):")
    logger.info("  1. Visit: https://github.com/j-hagedorn/trilogy/blob/master/data/aft.csv")
    logger.info("  2. Click 'Raw' to view the raw CSV")
    logger.info("  3. Right-click and 'Save as...'")
    logger.info("  4. Save to: data/western/aft.csv")


def print_gutenberg_links():
    """Print useful Project Gutenberg links for folktale collections"""
    print("\n" + "="*70)
    print("PROJECT GUTENBERG COLLECTIONS FOR ASIAN FOLKTALES")
    print("="*70)
    
    collections = {
        'Chinese Folktales': {
            'url': 'https://www.gutenberg.org/ebooks/search/?query=chinese+folktales',
            'authors': ['W. A. Clouston', 'James Legge'],
        },
        'Japanese Folktales': {
            'url': 'https://www.gutenberg.org/ebooks/search/?query=japanese+fairy+tales',
            'authors': ['Yei Theodora Ozaki'],
        },
        'Korean Tales': {
            'url': 'https://www.gutenberg.org/ebooks/search/?query=korean+tales',
            'authors': ['James Legge'],
        },
        'Vietnamese Folktales': {
            'url': 'https://www.gutenberg.org/ebooks/search/?query=vietnamese+folktales',
            'authors': [],
        },
        'Tibetan Folktales': {
            'url': 'https://www.gutenberg.org/ebooks/search/?query=tibetan+tales',
            'authors': [],
        },
        'Thai Folktales': {
            'url': 'https://www.gutenberg.org/ebooks/search/?query=thai+folktales',
            'authors': [],
        },
        'Indonesian/Malay Tales': {
            'url': 'https://www.gutenberg.org/ebooks/search/?query=malay+indonesian+folktales',
            'authors': ['Walter W. Skeat'],\n        },\n        'Philippine Folktales': {\n            'url': 'https://www.gutenberg.org/ebooks/search/?query=philippines+folktales',\n            'authors': [],\n        },\n    }\n    \n    for collection_name, info in collections.items():\n        print(f\"\\n{collection_name}:\")\n        print(f\"  URL: {info['url']}\")\n        if info['authors']:\n            print(f\"  Suggested authors: {', '.join(info['authors'])}\")\n        print(f\"  Instructions:\")\n        print(f\"    1. Search the URL above\")\n        print(f\"    2. Download as UTF-8 text (.txt)\")\n        print(f\"    3. Save to: data/asian/{collection_name.lower().replace(' ', '_')}/\")\n\n\ndef print_alternative_sources():\n    \"\"\"Print alternative data sources\"\"\"\n    print(\"\\n\" + \"=\"*70)\n    print(\"ALTERNATIVE DATA SOURCES\")\n    print(\"=\"*70)\n    \n    sources = {\n        'SurLaLune Database': {\n            'url': 'https://www.surlalunefairytales.com/',\n            'desc': 'Annotated folktale database with some Asian tales'\n        },\n        'Internet Sacred Text Archive': {\n            'url': 'https://sacred-texts.com/',\n            'desc': 'Religious and folklore texts (including Asian)'\n        },\n        'Folklore and Mythology Electronic Texts': {\n            'url': 'https://sites.psu.edu/psulibraries/2015/01/08/folklore-and-mythology-electronic-texts/',\n            'desc': 'University collection with global folklore'\n        },\n        'UCLA Folklore Library': {\n            'url': 'https://www.humnet.ucla.edu/humnet/data-resources-humanities-research/',\n            'desc': 'Academic folklore collection'\n        },\n        'Digital Public Library of America': {\n            'url': 'https://dp.la/',\n            'desc': 'Federated access to digital collections'\n        },\n        'National Geographic Folktales': {\n            'url': 'https://www.nationalgeographic.com/',\n            'desc': 'Cultural and folklore articles'\n        },\n    }\n    \n    for source_name, info in sources.items():\n        print(f\"\\n{source_name}:\")\n        print(f\"  Description: {info['desc']}\")\n        print(f\"  URL: {info['url']}\")\n\n\ndef print_data_organization_tips():\n    \"\"\"Print tips for organizing downloaded data\"\"\"\n    print(\"\\n\" + \"=\"*70)\n    print(\"DATA ORGANIZATION TIPS\")\n    print(\"=\"*70)\n    \n    print(\"\"\"\n1. **Naming Convention**:\n   - Use descriptive titles: \"<original_title>_<version>.txt\"\n   - Example: \"Journey_to_the_West_Ch1_v1.txt\"\n   - Avoid special characters in filenames\n\n2. **Encoding**:\n   - Save all files as UTF-8 (not ASCII or Windows-1252)\n   - This ensures proper handling of non-ASCII characters\n\n3. **File Cleanup**:\n   - Remove Project Gutenberg headers/footers if present\n   - Keep original story text intact\n   - Remove metadata/editor notes if they interfere with analysis\n\n4. **Batch Processing**:\n   - Use convert_encoding.py (if provided) to batch-convert files\n   - Verify a few files manually before processing all\n\n5. **Documentation**:\n   - Keep a manifest.csv in data/asian/ with:\n     - filename, title, region, source, source_url, download_date\n   - This helps track data provenance\n\nExample manifest.csv:\n```\nfilename,title,region,source,source_url,notes\njapan_heike_01.txt,Tales from the Heike,japanese,Project Gutenberg,https://...,Partial collection\nchina_journey_west.txt,Journey to the West,chinese,Internet Archive,https://...,Full text\n```\n    \"\"\")\n\n\ndef main():\n    \"\"\"Main function\"\"\"\n    import argparse\n    \n    parser = argparse.ArgumentParser(\n        description='Data collection helper for folktale classifier'\n    )\n    parser.add_argument(\n        'action',\n        nargs='?',\n        default='info',\n        choices=['setup', 'links', 'sources', 'tips', 'info'],\n        help='Action to perform'\n    )\n    \n    args = parser.parse_args()\n    \n    if args.action == 'setup':\n        create_data_directories()\n        logger.info(\"\\nNext steps:\")\n        logger.info(\"  1. Download Western folktales (aft.csv) from GitHub\")\n        logger.info(\"  2. Download Asian/SE Asian folktales from Project Gutenberg\")\n        logger.info(\"  3. Place files in appropriate data/ subdirectories\")\n    \n    elif args.action == 'links':\n        print_gutenberg_links()\n    \n    elif args.action == 'sources':\n        print_alternative_sources()\n    \n    elif args.action == 'tips':\n        print_data_organization_tips()\n    \n    elif args.action == 'info':\n        print(\"\\n\" + \"=\"*70)\n        print(\"FOLKTALE DATA COLLECTION HELPER\")\n        print(\"=\"*70)\n        print(\"\\nUsage: python data_collection_helper.py <action>\")\n        print(\"\\nActions:\")\n        print(\"  setup   - Create data directory structure\")\n        print(\"  links   - Print Project Gutenberg collection links\")\n        print(\"  sources - Print alternative data sources\")\n        print(\"  tips    - Print data organization best practices\")\n        print(\"  info    - Show this help message\")\n        print(\"\\nExample:\")\n        print(\"  python data_collection_helper.py setup\")\n        print(\"  python data_collection_helper.py links\")\n        print(\"\\nFor more info, see QUICKSTART.md\")\n    \n    print()\n\n\nif __name__ == '__main__':\n    main()
