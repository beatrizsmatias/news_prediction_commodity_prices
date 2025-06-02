import pandas as pd
import random
from datetime import timedelta
from datetime import datetime
import json
import random
import numpy as np
import re


def categorize_country(country):
    """Categorizes a text into country/countries based on known names, abbreviations, and major cities."""
    country_patterns = {
        # América do Norte
        'USA': r'\b(USA|U\.S\.A\.|United States|United States of America|America|Washington D\.C\.|New York|Houston|Texas|Permian Basin|American)\b',
        'Canada': r'\b(Canada|Canadian|Ottawa|Toronto|Vancouver|Alberta|Oil Sands|Montreal)\b',
        'Mexico': r'\b(Mexico|Mexican|Mexico City|Ciudad de México|Pemex|Mexicanos)\b',

        # Europa
        'UK': r'\b(UK|U\.K\.|United Kingdom|Britain|Great Britain|England|London|North Sea|British)\b',
        'Norway': r'\b(Norway|Norwegian|Oslo|North Sea)\b',
        'Russia': r'\b(Russia|Russian|Moscow|Siberia|Russian Federation|Kremlin|Gazprom|Rosneft)\b',
        'Germany': r'\b(Germany|German|Deutschland|Berlin|Bundesrepublik Deutschland)\b',
        'France': r'\b(France|French|Paris|République française|TotalEnergies)\b',
        'Italy': r'\b(Italy|Italian|Rome|Eni|Italia)\b',
        'Spain': r'\b(Spain|Spanish|Madrid|Barcelona|España)\b',

        # Oriente Médio
        'Saudi Arabia': r'\b(Saudi Arabia|Saudi|Riyadh|Aramco|Kingdom of Saudi Arabia|KSA)\b',
        'UAE': r'\b(UAE|United Arab Emirates|Abu Dhabi|Dubai|Emirati|ADNOC)\b',
        'Iran': r'\b(Iran|Tehran|Persian Gulf|Iranian|Islamic Republic of Iran)\b',
        'Iraq': r'\b(Iraq|Baghdad|Basra|Kirkuk|Iraqi)\b',
        'Kuwait': r'\b(Kuwait|Kuwaiti|Kuwait City)\b',
        'Qatar': r'\b(Qatar|Qatari|Doha)\b',
        'Oman': r'\b(Oman|Omani|Muscat)\b',

        # Ásia
        'China': r'\b(China|Chinese|Beijing|Shanghai|PRC|People\'s Republic of China|Sinopec|CNPC)\b',
        'India': r'\b(India|Indian|New Delhi|Mumbai|Hindustan|Bharat|ONGC|IOCL)\b',
        'Japan': r'\b(Japan|Japanese|Tokyo|JXTG|ENEOS|Nippon|Nihon)\b',
        'South Korea': r'\b(South Korea|Korea Republic|Seoul|Korean|Republic of Korea|ROK|SK Energy)\b',
        'Indonesia': r'\b(Indonesia|Indonesian|Jakarta|Pertamina)\b',

        # África
        'Nigeria': r'\b(Nigeria|Nigerian|Abuja|Lagos|NNPC)\b',
        'Angola': r'\b(Angola|Angolan|Luanda|Sonangol)\b',
        'Algeria': r'\b(Algeria|Algerian|Algiers|Sonatrach)\b',
        'Libya': r'\b(Libya|Libyan|Tripoli|Brega|NOC)\b',
        'South Africa': r'\b(South Africa|South African|Cape Town|Johannesburg|RSA|Republic of South Africa)\b',

        # América do Sul
        'Brazil': r'\b(Brazil|Brasília|São Paulo|Rio de Janeiro|Petrobras|Brazilian|Brasil)\b',
        'Venezuela': r'\b(Venezuela|Venezuelan|Caracas|PDVSA|Maracaibo)\b',
        'Argentina': r'\b(Argentina|Argentine|Argentinian|Buenos Aires|YPF)\b',
        'Colombia': r'\b(Colombia|Colombian|Bogotá|Ecopetrol)\b',
        'Ecuador': r'\b(Ecuador|Ecuadorian|Quito|Petroamazonas|Petroecuador)\b',

        # Oceania
        'Australia': r'\b(Australia|Australian|Canberra|Sydney|Melbourne|Aussie|Woodside|Santos)\b',

        # OPEP (menções adicionais)
        'OPEC': r'\b(OPEC|OPEP|Organization of the Petroleum Exporting Countries)\b',
    }

    matched_states = ""

    # Convert country to string to safely perform regex and substring checks
    country_str = str(country)

    i = 0
    for state, pattern in state_patterns.items():
        if re.search(pattern, country_str, re.IGNORECASE):
            if i == 0:
                matched_states += state
            else:
                matched_states += ',' + state
            i = i + 1

    if not matched_states:
        # América do Sul
        if any(term in country_str for term in ['South America', 'América do Sul', 'LatAm', 'Latin America', 'LATAM']):
            return 'South America'
        # América Central
        elif any(term in country_str for term in ['Central America', 'América Central', 'Centroamérica']):
            return 'Central America'
        # América do Norte
        elif any(term in country_str for term in ['North America', 'América do Norte', 'NAFTA']):
            return 'North America'
        # Europa
        elif any(term in country_str for term in ['Europe', 'European Union', 'EU', 'Europa']):
            return 'Europe'
        # Ásia
        elif any(term in country_str for term in ['Asia', 'Asian', 'Ásia', 'Southeast Asia', 'ASEAN']):
            return 'Asia'
        # Oriente Médio
        elif any(term in country_str for term in ['Middle East', 'Oriente Médio', 'Gulf States', 'GCC']):
            return 'Middle East'
        # África
        elif any(term in country_str for term in ['Africa', 'African', 'África', 'Sub-Saharan Africa', 'North Africa']):
            return 'Africa'
        # Oceania
        elif any(term in country_str for term in ['Oceania', 'Australasia', 'Pacific Islands']):
            return 'Oceania'
        # Nacional ou genérico
        elif any(term in country_str for term in ['Brazil', 'Brasil', 'National', 'General', 'multiple', 'Various']):
            return 'National'
        # Se nada for identificado
        else:
            return 'Global'
    return matched_states


def format_news(parsed_data, date, country):
    """Helper function to extract and format news from the parsed data."""
    parsed_data['categorized_country'] = parsed_data['country'].apply(categorize_country)
    parsed_data['date'] = pd.to_datetime(parsed_data['date'])
    # Format the date string to match the dataframe format
    date_str = date.strftime('%Y-%m-%d')

    # Filter news entries based on the date and the country
    filtered_news = parsed_data[
        (parsed_data['date'] == pd.to_datetime(date_str)) &
        ((parsed_data['country'].str.contains(country)) |
         (parsed_data['categorized_country'].isin(['Global', 'National'])))
    ]

    if not filtered_news.empty:
        news_texts = []
        for _, news_entry in filtered_news.iterrows():
            news_text = news_entry['news'].replace("..", ".")
            rationality = news_entry['rationality']
            time = news_entry['date'].strftime('%Y-%m-%d')
            news_texts.append(f"On {time}, in the state of {news_entry['categorized_country']}, the news was: '{news_text}'. Rationality behind it: {rationality}")
        return " ".join(news_texts)
    else:
        return f"No relevant news available for {date_str}."


def parse_news_TS_final(weather_data_file, news_data_file, ts_file, save_file):
    # all_weather_data = pd.read_csv(weather_data_file, encoding='utf-8')
    parsed_data = pd.read_csv(news_data_file, encoding='utf-8')

    # Ensure SETTLEMENTDATE column is of datetime type
    all_data = pd.read_csv(ts_file)
    all_data['SETTLEMENTDATE'] = pd.to_datetime(all_data['SETTLEMENTDATE'])

    # Initialize result list
    result_list = []

    # Get all unique country values
    unique_countrys = all_data['country'].unique()

    # Define time ranges (in days)
    time_ranges = {'day': 1}

    # Iterate through each country
    i = 0
    for country in unique_countrys:
        # Find and sort data for the current country
        country_data = all_data[all_data['country'] == country].sort_values(by='SETTLEMENTDATE')

        # Get possible start dates
        start_dates = pd.to_datetime(country_data['SETTLEMENTDATE'].dt.date.unique())

        # For each possible start date, randomly select a time range
        for start_date in start_dates:
            if start_date < pd.to_datetime("2019-1-1"):
                continue

            if start_date == pd.to_datetime("2021-1-1"):
                break

            # Randomly select a time range
            chosen_time_range = random.choice(list(time_ranges.values()))

            # Get the end date of the input period
            input_end_date = start_date + timedelta(days=chosen_time_range)

            if input_end_date.strftime('%Y-%m-%d') == "2023-11-30":
                print(input_end_date, " removed")
                continue

            # Get power data for the input period
            input_data = country_data[
                (country_data['SETTLEMENTDATE'] >= start_date) &
                (country_data['SETTLEMENTDATE'] < input_end_date)]

            if len(input_data['TOTALDEMAND'].tolist()) > chosen_time_range * 48:
                input_data.set_index('SETTLEMENTDATE', inplace=True)
                input_data = input_data['TOTALDEMAND'].resample('30T').mean().reset_index()

            # Get power data for the output period (next day)
            output_data = country_data[
                (country_data['SETTLEMENTDATE'] >= input_end_date) &
                (country_data['SETTLEMENTDATE'] < input_end_date + timedelta(days=1))]

            if len(output_data['TOTALDEMAND'].tolist()) > 48:
                output_data.set_index('SETTLEMENTDATE', inplace=True)
                output_data = output_data['TOTALDEMAND'].resample('30T').mean().reset_index()

            # If output_data is empty, skip this date
            if output_data.empty:
                continue

            # Get news data for the input and output periods
            new_parsed_data = parsed_data.copy()
            new_parsed_data["date"] = pd.to_datetime(new_parsed_data["date"])
            date_list = new_parsed_data["date"].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()

            if start_date.strftime('%Y-%m-%d') not in date_list:
                formatted_news_1 = ""
            else:
                formatted_news_1 = format_news(parsed_data, start_date, country[:-1])

            if input_end_date.strftime('%Y-%m-%d') not in date_list:
                formatted_news_2 = ""
            else:
                formatted_news_2 = format_news(parsed_data, input_end_date, country[:-1])

            # Format instruction and output, keeping one decimal point
            formatted_instruction = ",".join(f"{demand:.1f}" for demand in input_data['TOTALDEMAND'])
            formatted_output = ",".join(f"{demand:.1f}" for demand in output_data['TOTALDEMAND'])

            # Format input dates, removing leading zeros from months and days
            formatted_input_dates = ",".join(date.strftime('%m/%d %H:%M:%S').lstrip("0").replace('/0', '/') for date in input_data['SETTLEMENTDATE'])

            # Construct result dictionary
            result_dict = {
                "instruction": "The historical load data is: " + formatted_instruction,
                "input": "Based on the historical load data, please predict the load consumption in the next day. " +
                         "The country for prediction is " + country[:-1] + ". The start date of historical data was on " +
                         start_date.strftime('%Y-%m-%d').replace('-0', '-') +  " The data frequency is 30 minutes per point." + " Historical data covers " + str(
                    chosen_time_range) + " day." +
                         " The date of prediction is on " + input_end_date.strftime('%Y-%m-%d').replace('-0', '-') +
                         " that is " + check_weekday_or_weekend(input_end_date) +
                         formatted_news_1 + formatted_news_2,
                "output": formatted_output
            }
            print(result_dict)

            # Add to result list
            result_list.append(result_dict)
        i = i + 1
        print(f"country{i}: {country} is completed")

    # Display or save results
    with open(save_file, 'w') as f:
        json.dump(result_list, f)
    return result_list

def train_validation_split(json_file_path,train_save_file,val_save_file,val_num):
    # Reading the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:  # Use UTF-8 for reading
        result_list = json.load(file)
    print(result_list[-5:])
    
    #indices = np.load(random_indices_file).tolist()
    #test_dataset = [result_list[index] for index in indices]
    
    
    # Randomly selecting 2000 data points for the validation set
    #result_list = [data for data in result_list if data not in test_dataset]
    validation_set = random.sample(result_list, val_num)
    train_dataset = [data for data in result_list if data not in validation_set]

    #print(f"Test Set Size: {len(test_dataset)}")
    print(f"Validation Set Size: {len(validation_set)}")
    print(f"Train Dataset Size: {len(train_dataset)}")

    # Saving to files
    #with open(test_save_file, 'w', encoding='utf-8') as f:
        #json.dump(test_dataset, f, ensure_ascii=False, indent=4)

    with open(train_save_file, 'w', encoding='utf-8') as f_train:
        json.dump(train_dataset, f_train, ensure_ascii=False, indent=4)
        
    with open(val_save_file, 'w', encoding='utf-8') as f_val:
        json.dump(validation_set, f_val, ensure_ascii=False, indent=4)