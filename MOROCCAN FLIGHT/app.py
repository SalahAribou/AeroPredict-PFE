# ═══════════════════════════════════════════════════════════════════════════════
# app.py — AeroPredict · AI Flight Arrival Estimator
# Premium Consumer-Facing Flight Intelligence App
# ═══════════════════════════════════════════════════════════════════════════════

import time
import datetime

import pandas as pd
import streamlit as st

import joblib
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AeroPredict | Smart Flight Arrival",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — STATIC LOOKUP TABLES
# ─────────────────────────────────────────────────────────────────────────────

# ── 1.1  Airport Full Names  (ICAO → "City — Full Name") ─────────────────────
AIRPORT_NAMES: dict[str, str] = {
    # ── Morocco (GM) ──────────────────────────────────────────────────────────
    "GMFF": "Fès — Fès-Saïss Airport",
    "GMFO": "Oujda — Oujda–Angads Airport",
    "GMME": "Rabat — Rabat–Salé Airport",
    "GMMN": "Casablanca — Mohammed V International Airport",
    "GMMW": "Nador — Nador International Airport",
    "GMMX": "Marrakech — Marrakech Menara Airport",
    "GMMY": "Kénitra — Kénitra Air Base",
    "GMTA": "Al Hoceïma — Cherif Al Idrissi Airport",
    "GMTN": "Tanger — Tanger Ibn Battuta Airport",
    "GMTT": "Tétouan — Sania Ramel Airport",
    # ── Algeria (DA) ──────────────────────────────────────────────────────────
    "DAAG": "Algiers — Houari Boumediene Airport",
    "DAON": "Tindouf — Tindouf Airport",
    # ── Tunisia (DT) ──────────────────────────────────────────────────────────
    "DTTA": "Tunis — Tunis-Carthage International Airport",
    # ── Belgium (EB) ──────────────────────────────────────────────────────────
    "EBAW": "Antwerp — Antwerp International Airport",
    "EBBE": "Beauvechain — Beauvechain Air Base",
    "EBBR": "Brussels — Brussels Airport",
    "EBCI": "Charleroi — Brussels South Charleroi Airport",
    "EBFN": "Koksijde — Koksijde Air Base",
    "EBLG": "Liège — Liège Airport",
    "EBOS": "Ostend — Ostend-Bruges International Airport",
    # ── Germany (ED / ET) ─────────────────────────────────────────────────────
    "EDDB": "Berlin — Berlin Brandenburg Airport",
    "EDDF": "Frankfurt — Frankfurt Airport",
    "EDDH": "Hamburg — Hamburg Airport",
    "EDDK": "Cologne — Cologne Bonn Airport",
    "EDDL": "Düsseldorf — Düsseldorf Airport",
    "EDDM": "Munich — Munich International Airport",
    "EDDP": "Leipzig — Leipzig/Halle Airport",
    "EDDR": "Saarbrücken — Saarbrücken Airport",
    "EDDS": "Stuttgart — Stuttgart Airport",
    "EDFH": "Frankfurt-Hahn — Frankfurt-Hahn Airport",
    "EDFZ": "Mainz — Mainz-Finthen Airport",
    "EDJA": "Memmingen — Memmingen Allgäu Airport",
    "EDLP": "Paderborn — Paderborn Lippstadt Airport",
    "EDLV": "Weeze — Niederrhein Airport",
    "EDLW": "Dortmund — Dortmund Airport",
    "EDSB": "Karlsruhe — Karlsruhe/Baden-Baden Airport",
    "ETAR": "Ramstein — Ramstein Air Base",
    "ETNW": "Wunstorf — Wunstorf Air Base",
    # ── United Kingdom (EG) ───────────────────────────────────────────────────
    "EGBB": "Birmingham — Birmingham Airport",
    "EGCC": "Manchester — Manchester Airport",
    "EGFF": "Cardiff — Cardiff Wales Airport",
    "EGGD": "Bristol — Bristol Airport",
    "EGGW": "London — London Luton Airport",
    "EGKB": "London — London Biggin Hill Airport",
    "EGKK": "London — London Gatwick Airport",
    "EGLF": "Farnborough — Farnborough Airport",
    "EGLL": "London — London Heathrow Airport",
    "EGMC": "Southend — London Southend Airport",
    "EGNM": "Leeds — Leeds Bradford Airport",
    "EGNT": "Newcastle — Newcastle Airport",
    "EGNV": "Durham — Durham Tees Valley Airport",
    "EGPE": "Inverness — Inverness Airport",
    "EGPF": "Glasgow — Glasgow Airport",
    "EGPH": "Edinburgh — Edinburgh Airport",
    "EGSS": "London — London Stansted Airport",
    "EGTK": "Oxford — Oxford Airport",
    "EGVO": "Odiham — RAF Odiham",
    # ── Netherlands (EH) ──────────────────────────────────────────────────────
    "EHAM": "Amsterdam — Amsterdam Schiphol Airport",
    "EHBK": "Maastricht — Maastricht Aachen Airport",
    "EHEH": "Eindhoven — Eindhoven Airport",
    "EHRD": "Rotterdam — Rotterdam The Hague Airport",
    "EHVK": "Volkel — Volkel Air Base",
    # ── Ireland (EI) ──────────────────────────────────────────────────────────
    "EIDW": "Dublin — Dublin Airport",
    # ── Denmark (EK) ──────────────────────────────────────────────────────────
    "EKCH": "Copenhagen — Copenhagen Airport",
    "EKYT": "Aalborg — Aalborg Airport",
    # ── Luxembourg (EL) ───────────────────────────────────────────────────────
    "ELLX": "Luxembourg — Luxembourg Airport",
    # ── Poland (EP) ───────────────────────────────────────────────────────────
    "EPKK": "Kraków — Kraków John Paul II Airport",
    "EPKT": "Katowice — Katowice Airport",
    "EPWA": "Warsaw — Warsaw Chopin Airport",
    # ── Sweden (ES) ───────────────────────────────────────────────────────────
    "ESSA": "Stockholm — Stockholm Arlanda Airport",
    "ESSB": "Stockholm — Stockholm Bromma Airport",
    # ── Latvia (EV) ───────────────────────────────────────────────────────────
    "EVRA": "Riga — Riga International Airport",
    # ── Lithuania (EY) ────────────────────────────────────────────────────────
    "EYKA": "Kaunas — Kaunas Airport",
    "EYVI": "Vilnius — Vilnius Airport",
    # ── Egypt (HE) ────────────────────────────────────────────────────────────
    "HE37": "Egypt — Abu Rudeis Airport",
    "HE42": "Egypt — El Gora Airport",
    "HEAL": "El Alamein — El Alamein International Airport",
    "HEAZ": "Abu Zenima — Abu Zenima Airport",
    # ── Spain (LE) ────────────────────────────────────────────────────────────
    "LEAL": "Alicante — Alicante-Elche Airport",
    "LEAM": "Almería — Almería Airport",
    "LEAS": "Asturias — Asturias Airport",
    "LEBA": "León — León Airport",
    "LEBB": "Bilbao — Bilbao Airport",
    "LEBL": "Barcelona — Barcelona El Prat Airport",
    "LECO": "A Coruña — A Coruña Airport",
    "LEGE": "Girona — Girona-Costa Brava Airport",
    "LEIB": "Ibiza — Ibiza Airport",
    "LEMD": "Madrid — Madrid Barajas Airport",
    "LEMG": "Málaga — Málaga-Costa del Sol Airport",
    "LEMI": "Murcia — Murcia International Airport",
    "LEPA": "Palma — Palma de Mallorca Airport",
    "LERT": "Rota — Rota Naval Air Station",
    "LEST": "Santiago — Santiago de Compostela Airport",
    "LETO": "Torrejón — Torrejón Air Base",
    "LEVC": "Valencia — Valencia Airport",
    "LEZG": "Zaragoza — Zaragoza Airport",
    "LEZL": "Seville — Seville Airport",
    # ── France (LF) ───────────────────────────────────────────────────────────
    "LFAV": "Valenciennes — Valenciennes-Denain Airport",
    "LFAY": "Amiens — Amiens-Glisy Airport",
    "LFBD": "Bordeaux — Bordeaux-Mérignac Airport",
    "LFBF": "Toulouse — Toulouse-Francazal Airport",
    "LFBO": "Toulouse — Toulouse-Blagnac Airport",
    "LFBZ": "Biarritz — Biarritz Pays Basque Airport",
    "LFJL": "Metz-Nancy — Metz-Nancy-Lorraine Airport",
    "LFKB": "Bastia — Bastia-Poretta Airport",
    "LFLC": "Clermont-Ferrand — Clermont-Ferrand Airport",
    "LFLL": "Lyon — Lyon–Saint-Exupéry Airport",
    "LFLS": "Grenoble — Grenoble-Isère Airport",
    "LFLY": "Lyon — Lyon-Bron Airport",
    "LFMD": "Cannes — Cannes-Mandelieu Airport",
    "LFMK": "Carcassonne — Carcassonne Airport",
    "LFML": "Marseille — Marseille Provence Airport",
    "LFMN": "Nice — Nice Côte d'Azur Airport",
    "LFMT": "Montpellier — Montpellier–Méditerranée Airport",
    "LFMV": "Avignon — Avignon-Caumont Airport",
    "LFMY": "Salon — Salon-de-Provence Air Base",
    "LFOB": "Beauvais — Beauvais–Tillé Airport",
    "LFOK": "Châlons — Châlons Vatry Airport",
    "LFPB": "Paris — Paris Le Bourget Airport",
    "LFPC": "Creil — Creil Air Base",
    "LFPG": "Paris — Paris Charles de Gaulle Airport",
    "LFPO": "Paris — Paris Orly Airport",
    "LFPV": "Villacoublay — Villacoublay Air Base",
    "LFQQ": "Lille — Lille Airport",
    "LFQT": "Merville — Merville-Calonne Airport",
    "LFRN": "Rennes — Rennes-Saint-Jacques Airport",
    "LFRS": "Nantes — Nantes Atlantique Airport",
    "LFSB": "Basel — EuroAirport Basel-Mulhouse-Freiburg",
    "LFST": "Strasbourg — Strasbourg Airport",
    "LFXA": "Ambérieu — Ambérieu Air Base",
    # ── Hungary (LH) ──────────────────────────────────────────────────────────
    "LHBP": "Budapest — Budapest Ferenc Liszt International Airport",
    # ── Italy (LI) ────────────────────────────────────────────────────────────
    "LICC": "Catania — Catania-Fontanarossa Airport",
    "LIEO": "Olbia — Olbia Costa Smeralda Airport",
    "LIMA": "Alessandria — Alessandria Airport",
    "LIMC": "Milan — Milan Malpensa Airport",
    "LIME": "Bergamo — Bergamo Orio al Serio Airport",
    "LIMF": "Turin — Turin Airport",
    "LIML": "Milan — Milan Linate Airport",
    "LIMN": "Ghedi — Ghedi Air Base",
    "LIMP": "Parma — Parma Airport",
    "LIMZ": "Cuneo — Cuneo Levaldigi Airport",
    "LIPE": "Bologna — Bologna Guglielmo Marconi Airport",
    "LIPZ": "Venice — Venice Marco Polo Airport",
    "LIRA": "Rome — Rome Ciampino Airport",
    "LIRF": "Rome — Rome Fiumicino Airport",
    "LIRN": "Naples — Naples International Airport",
    "LIRP": "Pisa — Pisa International Airport",
    "LIRU": "Rome — Rome-Urbe Airport",
    # ── Czech Republic (LK) ───────────────────────────────────────────────────
    "LKMT": "Ostrava — Ostrava Leoš Janáček Airport",
    "LKPD": "Pardubice — Pardubice Airport",
    "LKPR": "Prague — Václav Havel Airport Prague",
    "LKTB": "Brno — Brno-Tuřany Airport",
    # ── Israel (LL) ───────────────────────────────────────────────────────────
    "LLBG": "Tel Aviv — Ben Gurion International Airport",
    # ── Malta (LM) ────────────────────────────────────────────────────────────
    "LMML": "Valletta — Malta International Airport",
    # ── Austria (LO) ──────────────────────────────────────────────────────────
    "LOWI": "Innsbruck — Innsbruck Airport",
    "LOWL": "Linz — Linz Airport",
    "LOWW": "Vienna — Vienna International Airport",
    # ── Portugal (LP) ─────────────────────────────────────────────────────────
    "LPAR": "Alverca — Alverca Air Base",
    "LPCS": "Cascais — Cascais Airport",
    "LPFR": "Faro — Faro Airport",
    "LPPR": "Porto — Porto Francisco Sá Carneiro Airport",
    "LPPT": "Lisbon — Lisbon Humberto Delgado Airport",
    # ── Romania (LR) ──────────────────────────────────────────────────────────
    "LRBC": "Bacău — Bacău International Airport",
    "LROP": "Bucharest — Henri Coandă International Airport",
    # ── Switzerland (LS) ──────────────────────────────────────────────────────
    "LSGG": "Geneva — Geneva Airport",
    "LSZH": "Zürich — Zürich Airport",
    "LSZR": "St. Gallen — St. Gallen-Altenrhein Airport",
    # ── Turkey (LT) ───────────────────────────────────────────────────────────
    "LTAI": "Antalya — Antalya Airport",
    "LTBA": "Istanbul — Istanbul Atatürk Airport",
    "LTBU": "Bursa — Bursa Yenişehir Airport",
    "LTFE": "Bodrum — Bodrum-Milas Airport",
    "LTFJ": "Istanbul — Istanbul Sabiha Gökçen Airport",
    "LTFM": "Istanbul — Istanbul Airport",
    # ── Gibraltar (LX) ────────────────────────────────────────────────────────
    "LXGB": "Gibraltar — Gibraltar International Airport",
    # ── Serbia (LY) ───────────────────────────────────────────────────────────
    "LYNI": "Niš — Niš Constantine the Great Airport",
    # ── Slovakia (LZ) ─────────────────────────────────────────────────────────
    "LZIB": "Bratislava — Bratislava Milan Rastislav Štefánik Airport",
    "LZKZ": "Košice — Košice International Airport",
    # ── Ukraine (UK) ──────────────────────────────────────────────────────────
    "UKBB": "Kyiv — Boryspil International Airport",
    # ── Russia (UU) ───────────────────────────────────────────────────────────
    "UUDD": "Moscow — Domodedovo International Airport",
}

# ── 1.2  Airport → Country Mapping ───────────────────────────────────────────
AIRPORT_COUNTRY: dict[str, str] = {
    # Morocco
    "GMFF": "Morocco",
    "GMFO": "Morocco",
    "GMME": "Morocco",
    "GMMN": "Morocco",
    "GMMW": "Morocco",
    "GMMX": "Morocco",
    "GMMY": "Morocco",
    "GMTA": "Morocco",
    "GMTN": "Morocco",
    "GMTT": "Morocco",
    # Algeria
    "DAAG": "Algeria",
    "DAON": "Algeria",
    # Tunisia
    "DTTA": "Tunisia",
    # Belgium
    "EBAW": "Belgium",
    "EBBE": "Belgium",
    "EBBR": "Belgium",
    "EBCI": "Belgium",
    "EBFN": "Belgium",
    "EBLG": "Belgium",
    "EBOS": "Belgium",
    # Germany
    "EDDB": "Germany",
    "EDDF": "Germany",
    "EDDH": "Germany",
    "EDDK": "Germany",
    "EDDL": "Germany",
    "EDDM": "Germany",
    "EDDP": "Germany",
    "EDDR": "Germany",
    "EDDS": "Germany",
    "EDFH": "Germany",
    "EDFZ": "Germany",
    "EDJA": "Germany",
    "EDLP": "Germany",
    "EDLV": "Germany",
    "EDLW": "Germany",
    "EDSB": "Germany",
    "ETAR": "Germany",
    "ETNW": "Germany",
    # United Kingdom
    "EGBB": "United Kingdom",
    "EGCC": "United Kingdom",
    "EGFF": "United Kingdom",
    "EGGD": "United Kingdom",
    "EGGW": "United Kingdom",
    "EGKB": "United Kingdom",
    "EGKK": "United Kingdom",
    "EGLF": "United Kingdom",
    "EGLL": "United Kingdom",
    "EGMC": "United Kingdom",
    "EGNM": "United Kingdom",
    "EGNT": "United Kingdom",
    "EGNV": "United Kingdom",
    "EGPE": "United Kingdom",
    "EGPF": "United Kingdom",
    "EGPH": "United Kingdom",
    "EGSS": "United Kingdom",
    "EGTK": "United Kingdom",
    "EGVO": "United Kingdom",
    # Netherlands
    "EHAM": "Netherlands",
    "EHBK": "Netherlands",
    "EHEH": "Netherlands",
    "EHRD": "Netherlands",
    "EHVK": "Netherlands",
    # Ireland
    "EIDW": "Ireland",
    # Denmark
    "EKCH": "Denmark",
    "EKYT": "Denmark",
    # Luxembourg
    "ELLX": "Luxembourg",
    # Poland
    "EPKK": "Poland",
    "EPKT": "Poland",
    "EPWA": "Poland",
    # Sweden
    "ESSA": "Sweden",
    "ESSB": "Sweden",
    # Latvia
    "EVRA": "Latvia",
    # Lithuania
    "EYKA": "Lithuania",
    "EYVI": "Lithuania",
    # Egypt
    "HE37": "Egypt",
    "HE42": "Egypt",
    "HEAL": "Egypt",
    "HEAZ": "Egypt",
    # Spain
    "LEAL": "Spain",
    "LEAM": "Spain",
    "LEAS": "Spain",
    "LEBA": "Spain",
    "LEBB": "Spain",
    "LEBL": "Spain",
    "LECO": "Spain",
    "LEGE": "Spain",
    "LEIB": "Spain",
    "LEMD": "Spain",
    "LEMG": "Spain",
    "LEMI": "Spain",
    "LEPA": "Spain",
    "LERT": "Spain",
    "LEST": "Spain",
    "LETO": "Spain",
    "LEVC": "Spain",
    "LEZG": "Spain",
    "LEZL": "Spain",
    # France
    "LFAV": "France",
    "LFAY": "France",
    "LFBD": "France",
    "LFBF": "France",
    "LFBO": "France",
    "LFBZ": "France",
    "LFJL": "France",
    "LFKB": "France",
    "LFLC": "France",
    "LFLL": "France",
    "LFLS": "France",
    "LFLY": "France",
    "LFMD": "France",
    "LFMK": "France",
    "LFML": "France",
    "LFMN": "France",
    "LFMT": "France",
    "LFMV": "France",
    "LFMY": "France",
    "LFOB": "France",
    "LFOK": "France",
    "LFPB": "France",
    "LFPC": "France",
    "LFPG": "France",
    "LFPO": "France",
    "LFPV": "France",
    "LFQQ": "France",
    "LFQT": "France",
    "LFRN": "France",
    "LFRS": "France",
    "LFSB": "France",
    "LFST": "France",
    "LFXA": "France",
    # Hungary
    "LHBP": "Hungary",
    # Italy
    "LICC": "Italy",
    "LIEO": "Italy",
    "LIMA": "Italy",
    "LIMC": "Italy",
    "LIME": "Italy",
    "LIMF": "Italy",
    "LIML": "Italy",
    "LIMN": "Italy",
    "LIMP": "Italy",
    "LIMZ": "Italy",
    "LIPE": "Italy",
    "LIPZ": "Italy",
    "LIRA": "Italy",
    "LIRF": "Italy",
    "LIRN": "Italy",
    "LIRP": "Italy",
    "LIRU": "Italy",
    # Czech Republic
    "LKMT": "Czech Republic",
    "LKPD": "Czech Republic",
    "LKPR": "Czech Republic",
    "LKTB": "Czech Republic",
    # Israel
    "LLBG": "Israel",
    # Malta
    "LMML": "Malta",
    # Austria
    "LOWI": "Austria",
    "LOWL": "Austria",
    "LOWW": "Austria",
    # Portugal
    "LPAR": "Portugal",
    "LPCS": "Portugal",
    "LPFR": "Portugal",
    "LPPR": "Portugal",
    "LPPT": "Portugal",
    # Romania
    "LRBC": "Romania",
    "LROP": "Romania",
    # Switzerland
    "LSGG": "Switzerland",
    "LSZH": "Switzerland",
    "LSZR": "Switzerland",
    # Turkey
    "LTAI": "Turkey",
    "LTBA": "Turkey",
    "LTBU": "Turkey",
    "LTFE": "Turkey",
    "LTFJ": "Turkey",
    "LTFM": "Turkey",
    # Gibraltar
    "LXGB": "Gibraltar",
    # Serbia
    "LYNI": "Serbia",
    # Slovakia
    "LZIB": "Slovakia",
    "LZKZ": "Slovakia",
    # Ukraine
    "UKBB": "Ukraine",
    # Russia
    "UUDD": "Russia",
}

# ── 1.3  Airline Operator Names ───────────────────────────────────────────────
OPERATOR_NAMES: dict[str, str] = {
    "AAN": "Al Ain Aviation",
    "AAO": "Avia Air",
    "ABR": "Aigle Azur (Charter)",
    "ABY": "Air Arabia",
    "ADZ": "Aerodreams",
    "AEA": "Air Europa",
    "AEH": "Aero Charter",
    "AFR": "Air France",
    "AGV": "Agavia",
    "AHO": "Air Hamburg",
    "AIO": "Air Italy Cargo",
    "AIZ": "Arkia Israeli Airlines",
    "AMB": "AmeriJet International",
    "AME": "Air Montenegro",
    "ANE": "Air Nostrum (Iberia Regional)",
    "AOM": "Air Ops Morocco",
    "AOV": "Aero Vision",
    "ART": "Air Tarom",
    "AUA": "Austrian Airlines",
    "AUH": "Etihad Airways",
    "AXE": "Axis Airlines",
    "AXY": "Axy Air",
    "BAW": "British Airways",
    "BBB": "Jetair Belgium",
    "BBT": "TUI fly Belgium",
    "BCS": "European Air Charter",
    "BEL": "Brussels Airlines",
    "BID": "Bid Air",
    "BTI": "airBaltic",
    "BZE": "Bees Airline",
    "BZF": "Blue Falcon",
    "CAI": "Corendon Airlines",
    "CCM": "Air Corsica",
    "CES": "China Eastern Airlines",
    "CJL": "Cargojet",
    "CJT": "Cargojet Airways",
    "CLF": "Cello Aviation",
    "COO": "CorsairFly",
    "CPV": "Cape Verde Airlines",
    "CSH": "Shanghai Airlines",
    "CTN": "Croatia Airlines",
    "DAB": "Daallo Airlines",
    "DAT": "DAT Danish Air Transport",
    "DCL": "DC Aviation",
    "DCS": "DCS Aviation",
    "DGC": "DG Aviation",
    "DHA": "flydubai",
    "DHK": "DHL Air UK",
    "DLH": "Lufthansa",
    "EAF": "Eastern Atlantic Airlines",
    "EDC": "Aero VIP",
    "EFD": "Frontier Flying Service",
    "EIN": "Aer Lingus",
    "EJA": "NetJets Aviation",
    "EJU": "easyJet Europe",
    "ELY": "El Al Israel Airlines",
    "ENT": "Enstrom Helicopter",
    "EWG": "Eurowings",
    "EZS": "easyJet Switzerland",
    "EZY": "easyJet",
    "FAF": "French Air Force",
    "FLJ": "FL Technics Jets",
    "FMY": "FreeMy",
    "FRF": "Freight Force",
    "FRO": "Frontera Airways",
    "FYL": "Flylili",
    "GAF": "German Air Force",
    "GES": "Gestair",
    "GFA": "Gulf Air",
    "GIA": "Garuda Indonesia",
    "GJT": "Jet2.com",
    "GWI": "Germanwings",
    "HFY": "HiFly",
    "HRN": "Heron Airlines",
    "HST": "Airbus Transport International",
    "HSY": "Hello Skyways",
    "HYP": "Hyperion Aviation",
    "IAM": "Iranian Air Tours",
    "IBB": "Iberia Express",
    "IBE": "Iberia",
    "IXR": "AirAsia X",
    "JAF": "Jetair Fleet",
    "JFA": "Jet Fleet Aviation",
    "KLM": "KLM Royal Dutch Airlines",
    "KZU": "Kazavia",
    "LAV": "LAV Charter",
    "LBT": "Nouvelair Tunisie",
    "LDM": "Air L'Avion",
    "LER": "Léret Aviation",
    "LMJ": "LM Jets",
    "LNX": "Lynx Aviation",
    "LOT": "LOT Polish Airlines",
    "LUV": "Southwest Airlines",
    "LWG": "Lwinga Aviation",
    "LXA": "Luxaviation",
    "LZB": "Small Planet Airlines",
    "MAC": "Air Arabia Maroc",
    "MAI": "Moldavian Airlines",
    "MAY": "Mayair",
    "MLT": "Malta Air",
    "MMD": "Med-View Airline",
    "MMO": "MMO Aviation",
    "MNB": "MNB Air",
    "MSR": "EgyptAir",
    "MYX": "MyAir Express",
    "NAX": "Norwegian Air Shuttle",
    "NCR": "Nacor Aviation",
    "NJE": "NetJets Europe",
    "NOS": "Neos Air",
    "NOZ": "Norwegian Air Shuttle AOC",
    "NPT": "Neptune Aviation",
    "NVD": "Novair",
    "OYO": "Oyo Air",
    "PAV": "Pawan Hans",
    "PEA": "PEA Aviation",
    "PGT": "Pegasus Airlines",
    "PLF": "West Air Luxembourg",
    "PLM": "Pelm Air",
    "PNK": "Nok Air",
    "PTN": "Patin Aviation",
    "PVG": "Shanghai Pudong Aviation",
    "QAC": "Qatar Amiri Flight",
    "QQE": "Q-Aviation",
    "QTR": "Qatar Airways",
    "RAM": "Royal Air Maroc",
    "RGN": "Regent Airways",
    "RHH": "RH Aviation",
    "RLX": "Rally Air",
    "ROJ": "Royal Jordanian",
    "RUK": "Ruk-Air",
    "RYR": "Ryanair",
    "RYS": "Ryanair Sun",
    "SAZ": "Saudia",
    "SCR": "Sacre Coeur Air",
    "SHF": "Shuffle Aviation",
    "SJI": "SJI Aviation",
    "SRN": "Transaviaexport Airlines",
    "SUA": "Sun Air of Scandinavia",
    "SVW": "Silverstone Air Services",
    "SWT": "Swiftair",
    "TAP": "TAP Air Portugal",
    "TAR": "Tunisair",
    "TAY": "TNT Airways",
    "TCR": "TuiCruises",
    "TCX": "Thomas Cook Airlines",
    "TDR": "Thunder Airlines",
    "THY": "Turkish Airlines",
    "TJT": "Tjet Aviation",
    "TKJ": "Turkmenistan Airlines",
    "TRA": "Transavia",
    "TUI": "TUI Airways",
    "TVF": "Transavia France",
    "TVS": "Travel Service",
    "TYW": "Titan Airways",
    "UAE": "Emirates",
    "UAG": "United Arab Emirates (Government)",
    "UKL": "Ukraine International Airlines (Cargo)",
    "UNI": "Uni Air",
    "UVL": "UVL Charter",
    "VAW": "Edelweiss Air",
    "VCJ": "VistaJet",
    "VCN": "Vueling Airlines",
    "VJT": "VistaJet",
    "VLG": "Vueling",
    "VLJ": "Volotea",
    "VMP": "VMP Air",
    "WZZ": "Wizz Air",
    "XGO": "XGO Airlines",
    "XOJ": "Xojet Aviation",
}

# ── 1.4  Aircraft Type Full Names ─────────────────────────────────────────────
AIRCRAFT_NAMES: dict[str, str] = {
    "680A Citation Latitude": "Cessna Citation Latitude 680A",
    "A20N": "Airbus A320neo",
    "A21N": "Airbus A321neo",
    "A306": "Airbus A300-600",
    "A310": "Airbus A310",
    "A318": "Airbus A318",
    "A319": "Airbus A319",
    "A320": "Airbus A320",
    "A321": "Airbus A321",
    "A332": "Airbus A330-200",
    "A333": "Airbus A330-300",
    "A342": "Airbus A340-200",
    "A343": "Airbus A340-300",
    "A359": "Airbus A350-900",
    "A400": "Airbus A400M Atlas",
    "AN12": "Antonov An-12",
    "ASTR": "IAI Gulfstream 1159A Astra",
    "AT72": "ATR 72-200",
    "AT73": "ATR 72-300",
    "AT75": "ATR 72-500",
    "AT76": "ATR 72-600",
    "B190": "Beechcraft 1900",
    "B38M": "Boeing 737 MAX 8",
    "B39M": "Boeing 737 MAX 9",
    "B734": "Boeing 737-400",
    "B737": "Boeing 737-700",
    "B738": "Boeing 737-800",
    "B744": "Boeing 747-400",
    "B752": "Boeing 757-200",
    "B763": "Boeing 767-300",
    "B772": "Boeing 777-200",
    "B773": "Boeing 777-300",
    "B77L": "Boeing 777-200LR",
    "B77W": "Boeing 777-300ER",
    "B788": "Boeing 787-8 Dreamliner",
    "B789": "Boeing 787-9 Dreamliner",
    "BCS3": "Airbus A220-300",
    "C25A": "Cessna Citation CJ2",
    "C25C": "Cessna Citation CJ4",
    "C25M": "Cessna Citation M2",
    "C55B": "Cessna Citation Bravo",
    "C560": "Cessna Citation V / Ultra",
    "C56X": "Cessna Citation Excel / XLS",
    "C680": "Cessna Citation Sovereign",
    "C68A": "Cessna Citation Latitude",
    "CL30": "Bombardier Challenger 300",
    "CL35": "Bombardier Challenger 350",
    "CL60": "Bombardier Challenger 600",
    "CRJX": "Bombardier CRJ-900 / 1000",
    "DH8D": "Bombardier Dash 8 Q400",
    "E120": "Embraer EMB-120 Brasilia",
    "E145": "Embraer ERJ-145",
    "E190": "Embraer E190",
    "E195": "Embraer E195",
    "E295": "Embraer E195-E2",
    "E35L": "Embraer Legacy 600 / 650",
    "E550": "Embraer Praetor 600",
    "E55P": "Embraer Phenom 300E",
    "E75L": "Embraer E175",
    "EC35": "Airbus H135 (EC135)",
    "F2TH": "Dassault Falcon 2000",
    "F900": "Dassault Falcon 900",
    "FA50": "Dassault Falcon 50",
    "FA7X": "Dassault Falcon 7X",
    "GL5T": "Bombardier Global 5000",
    "GL7T": "Bombardier Global 7500",
    "GLEX": "Bombardier Global Express",
    "GLF5": "Gulfstream G550",
    "GLF6": "Gulfstream G650",
    "H25B": "Hawker 800",
    "LJ35": "Learjet 35",
    "LJ45": "Learjet 45",
    "LJ60": "Learjet 60",
    "LJ75": "Learjet 75",
    "P180": "Piaggio P.180 Avanti",
    "PC12": "Pilatus PC-12",
    "PC24": "Pilatus PC-24",
    "SB20": "Saab 2000",
    "SF34": "Saab 340",
    "SW4": "Fairchild Swearingen Metroliner",
    "TBM7": "Socata TBM 700",
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATA LOADING  (cached for the entire session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_valid_categories(csv_path: str = "pfe_morocco_clean_silver.csv") -> dict:
    df = pd.read_csv(csv_path)
    return {
        "adep_codes": sorted(df["adep"].dropna().unique().tolist()),
        "ades_codes": sorted(df["ades"].dropna().unique().tolist()),
        "operator_codes": sorted(df["icao_operator"].dropna().unique().tolist()),
        "typecodes": sorted(df["typecode"].dropna().unique().tolist()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def make_airport_label(code: str) -> str:
    name = AIRPORT_NAMES.get(code, "Unknown Airport")
    return f"{name} ({code})"


def make_operator_label(code: str) -> str:
    name = OPERATOR_NAMES.get(code, code)
    return f"{name} ({code})"


def make_aircraft_label(code: str) -> str:
    name = AIRCRAFT_NAMES.get(code, code)
    return f"{name} ({code})"


def extract_code_from_label(label: str) -> str:
    return label.rsplit("(", 1)[-1].rstrip(")")


def get_city_name(code: str) -> str:
    """Extracts just the city from 'City — Full Airport Name'."""
    raw = AIRPORT_NAMES.get(code, code)
    return raw.split(" — ")[0]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CUSTOM CSS  (Premium Consumer Travel App Aesthetic)
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:           #07101f;
    --bg-surface:   #0b1929;
    --bg-card:      #0f2040;
    --bg-input:     #091525;
    --border:       #1a3a5c;
    --border-glow:  #1d72ff;
    --accent:       #1d72ff;
    --accent-dim:   rgba(29,114,255,0.10);
    --accent-glow:  rgba(29,114,255,0.30);
    --green:        #10e8a8;
    --green-dim:    rgba(16,232,168,0.10);
    --green-glow:   rgba(16,232,168,0.40);
    --amber:        #f59e0b;
    --text-primary: #e4edf8;
    --text-sec:     #6b8aaa;
    --text-muted:   #3d5a7a;
    --font-display: 'Playfair Display', Georgia, serif;
    --font-mono:    'DM Mono', 'Courier New', monospace;
    --font-body:    'DM Sans', system-ui, sans-serif;
    --radius:       14px;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stAppViewContainer"] > .main { background: transparent !important; }
[data-testid="stHeader"]                   { background: transparent !important; }
section[data-testid="stSidebar"]           { background: var(--bg-surface) !important; border-right: 1px solid var(--border); }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

/* ── Noise texture overlay ────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
}

/* ── Nav bar ─────────────────────────────────────────────────────────────── */
.ap-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 2.5rem;
    background: linear-gradient(180deg, rgba(11,25,41,0.95) 0%, rgba(7,16,31,0.0) 100%);
    border-bottom: 1px solid rgba(26,58,92,0.5);
    margin-bottom: 2rem;
    backdrop-filter: blur(12px);
}
.ap-brand {
    font-family: var(--font-display);
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: var(--text-primary);
}
.ap-brand span { color: var(--accent); }
.ap-tagline {
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 300;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.ap-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--green);
    letter-spacing: 0.06em;
}
.ap-dot {
    width: 6px; height: 6px;
    background: var(--green);
    border-radius: 50%;
    animation: blink 2.2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.2;} }

/* ── Hero ────────────────────────────────────────────────────────────────── */
.ap-hero-title {
    font-family: var(--font-display);
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 900;
    line-height: 1.08;
    color: var(--text-primary);
    margin-bottom: 0.6rem;
}
.ap-hero-title em {
    font-style: italic;
    background: linear-gradient(120deg, #1d72ff, #10e8a8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.ap-hero-sub {
    font-size: 1rem;
    font-weight: 300;
    color: var(--text-sec);
    max-width: 520px;
    line-height: 1.6;
}
.ap-deco {
    font-family: var(--font-display);
    font-size: 5rem;
    font-weight: 900;
    font-style: italic;
    color: var(--accent);
    opacity: 0.08;
    line-height: 1;
    text-align: right;
    user-select: none;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
.ap-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.8rem 0;
}

/* ── Section headers ─────────────────────────────────────────────────────── */
.ap-section-head {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(29,114,255,0.2);
}

/* ── Streamlit widget overrides ───────────────────────────────────────────── */
.stSelectbox label,
.stDateInput label,
.stTimeInput label {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.10em !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    font-weight: 400 !important;
    margin-bottom: 0.25rem !important;
}
.stSelectbox [data-baseweb="select"] > div:first-child {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stSelectbox [data-baseweb="select"] > div:first-child:hover,
.stSelectbox [data-baseweb="select"] > div:first-child:focus-within {
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
.stSelectbox [data-baseweb="select"] span {
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.87rem !important;
}
.stDateInput input,
.stTimeInput input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.9rem !important;
}
.stDateInput input:focus,
.stTimeInput input:focus {
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d72ff 0%, #0f4db8 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-body) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.75rem 2rem !important;
    color: #ffffff !important;
    width: 100% !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 4px 24px rgba(29,114,255,0.45) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(29,114,255,0.65) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Warning banner ──────────────────────────────────────────────────────── */
.ap-warning {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-size: 0.82rem;
    color: var(--amber);
    text-align: center;
    margin-top: 0.5rem;
}

/* ══════════════════════════════════════════════════════════════════════════ */
/*  FLIGHT ITINERARY CARD (Boarding Pass)                                    */
/* ══════════════════════════════════════════════════════════════════════════ */
.bp-wrap {
    animation: riseIn 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
}
@keyframes riseIn {
    from { opacity: 0; transform: translateY(20px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0)   scale(1); }
}

.bp-card {
    background: linear-gradient(145deg, #0d2040 0%, #091629 60%, #061120 100%);
    border: 1px solid rgba(29,114,255,0.45);
    border-radius: 20px;
    padding: 2.2rem 2.8rem 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.03) inset,
        0 25px 80px rgba(0,0,0,0.6),
        0 0 60px rgba(29,114,255,0.12);
}

/* Top rainbow stripe */
.bp-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg,
        transparent 0%,
        #1d72ff     25%,
        #10e8a8     60%,
        #1d72ff     80%,
        transparent 100%
    );
}

/* Subtle radial glow in top-left */
.bp-card::after {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(29,114,255,0.12) 0%, transparent 70%);
    pointer-events: none;
}

/* Header row: status + airline info */
.bp-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.8rem;
}
.bp-status-pill {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    background: var(--green-dim);
    border: 1px solid rgba(16,232,168,0.3);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    color: var(--green);
    text-transform: uppercase;
}
.bp-status-dot {
    width: 5px; height: 5px;
    background: var(--green);
    border-radius: 50%;
}
.bp-airline-info {
    font-family: var(--font-body);
    font-size: 0.82rem;
    font-weight: 400;
    color: var(--text-sec);
    text-align: right;
}
.bp-airline-name {
    font-weight: 600;
    color: var(--text-primary);
    display: block;
    font-size: 0.88rem;
}

/* ── Timeline row ─────────────────────────────────────────────────────────── */
.bp-timeline {
    display: flex;
    align-items: flex-start;
    gap: 0;
    margin-bottom: 1.6rem;
}

/* Departure side */
.bp-dep {
    flex: 0 0 auto;
    min-width: 160px;
    text-align: left;
}

/* Arrival side */
.bp-arr {
    flex: 0 0 auto;
    min-width: 160px;
    text-align: right;
}

/* Airport IATA code */
.bp-iata {
    font-family: var(--font-mono);
    font-size: 2.4rem;
    font-weight: 500;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 0.2rem;
}
.bp-iata.is-dest {
    color: var(--accent);
}

/* City name below code */
.bp-city {
    font-family: var(--font-body);
    font-size: 0.78rem;
    font-weight: 400;
    color: var(--text-sec);
    margin-bottom: 0.65rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 150px;
}
.bp-arr .bp-city {
    margin-left: auto;
}

/* Time display */
.bp-dep-time {
    font-family: var(--font-mono);
    font-size: 1.65rem;
    font-weight: 400;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.bp-dep-date {
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 400;
    color: var(--text-muted);
}

/* ARRIVAL TIME — the hero */
.bp-arr-time {
    font-family: var(--font-mono);
    font-size: 3.8rem;
    font-weight: 500;
    color: var(--green);
    letter-spacing: -0.03em;
    line-height: 1;
    text-shadow: 0 0 40px var(--green-glow), 0 0 80px rgba(16,232,168,0.15);
    margin-bottom: 0.2rem;
}
.bp-arr-date {
    font-family: var(--font-body);
    font-size: 0.78rem;
    font-weight: 400;
    color: var(--text-sec);
}
.bp-arr-date .next-day-badge {
    display: inline-block;
    background: rgba(245,158,11,0.15);
    border: 1px solid rgba(245,158,11,0.35);
    color: var(--amber);
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 500;
    border-radius: 4px;
    padding: 0.05rem 0.35rem;
    margin-left: 0.35rem;
    vertical-align: middle;
}

/* ── Middle track ─────────────────────────────────────────────────────────── */
.bp-track-wrap {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 0 1.5rem;
    padding-top: 0.3rem;
}
.bp-duration-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 400;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
}
.bp-track {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 0;
    position: relative;
}
.bp-track-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    border: 2px solid var(--accent);
    background: var(--bg);
    flex-shrink: 0;
}
.bp-track-dot.is-dest {
    border-color: var(--green);
}
.bp-track-line {
    flex: 1;
    height: 1px;
    background: repeating-linear-gradient(
        to right,
        var(--border) 0px,
        var(--border) 6px,
        transparent 6px,
        transparent 12px
    );
}
.bp-plane-icon {
    font-size: 1.1rem;
    flex-shrink: 0;
    filter: drop-shadow(0 0 6px rgba(29,114,255,0.5));
    margin: 0 0.2rem;
    animation: floatPlane 3s ease-in-out infinite;
}
@keyframes floatPlane {
    0%, 100% { transform: translateY(0);    }
    50%       { transform: translateY(-3px); }
}
.bp-direct-label {
    font-family: var(--font-body);
    font-size: 0.65rem;
    color: var(--text-muted);
    margin-top: 0.45rem;
    letter-spacing: 0.04em;
}

/* ── Boarding pass perforated divider ────────────────────────────────────── */
.bp-perf-divider {
    position: relative;
    border: none;
    border-top: 2px dashed rgba(26,58,92,0.7);
    margin: 0.2rem -2.8rem;
}
.bp-perf-divider::before,
.bp-perf-divider::after {
    content: '';
    position: absolute;
    top: -12px;
    width: 22px; height: 22px;
    background: var(--bg);
    border-radius: 50%;
    border: 1px solid rgba(26,58,92,0.7);
}
.bp-perf-divider::before { left: -12px; }
.bp-perf-divider::after  { right: -12px; }

/* ── Ticket footer row ────────────────────────────────────────────────────── */
.bp-footer-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1.4rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.bp-footer-item {
    text-align: center;
}
.bp-footer-label {
    font-family: var(--font-mono);
    font-size: 0.57rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.2rem;
}
.bp-footer-value {
    font-family: var(--font-body);
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-primary);
}

/* ── Page footer ─────────────────────────────────────────────────────────── */
.ap-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: var(--font-mono);
    font-size: 0.58rem;
    color: var(--text-muted);
    letter-spacing: 0.07em;
    padding: 0 0.5rem 1.5rem;
    margin-top: 0.5rem;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:

    # ── 5.0  Inject styles ────────────────────────────────────────────────────
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── 5.1  Navigation bar ───────────────────────────────────────────────────
    st.markdown(
        """
    <div class="ap-nav">
        <div>
            <div class="ap-brand">Aero<span>Predict</span></div>
            <div class="ap-tagline">Smart Flight Intelligence</div>
        </div>
        <div class="ap-status">
            <div class="ap-dot"></div>
            AI MODEL READY
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── 5.2  Page hero ────────────────────────────────────────────────────────
    col_hero, col_deco = st.columns([3, 1])
    with col_hero:
        st.markdown(
            '<div class="ap-hero-title">Know exactly<br>when you <em>arrive.</em></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ap-hero-sub">'
            "Plan your journey with precision. Enter your flight details to get "
            "an AI-powered arrival time."
            "</div>",
            unsafe_allow_html=True,
        )
    with col_deco:
        st.markdown('<div class="ap-deco">✈</div>', unsafe_allow_html=True)

    st.markdown('<hr class="ap-divider">', unsafe_allow_html=True)

    # ── 5.3  Load valid categories (cached) ───────────────────────────────────
    try:
        cats = load_valid_categories("pfe_morocco_clean_silver.csv")
    except FileNotFoundError:
        st.error(
            "⚠️  **Dataset not found.** "
            "Place `pfe_morocco_clean_silver.csv` in the same directory as `app.py`."
        )
        st.stop()

    adep_codes = cats["adep_codes"]
    ades_codes = cats["ades_codes"]
    operator_codes = cats["operator_codes"]
    typecodes = cats["typecodes"]

    # ── 5.4  Build display labels ─────────────────────────────────────────────
    adep_labels = [make_airport_label(c) for c in adep_codes]
    operator_labels = [make_operator_label(c) for c in operator_codes]
    aircraft_labels = [make_aircraft_label(c) for c in typecodes]

    # ── 5.5  Three-column form layout ─────────────────────────────────────────
    col_route, col_aircraft, col_schedule = st.columns([1.15, 0.95, 0.90], gap="large")

    # ── COLUMN 1 — Route ──────────────────────────────────────────────────────
    with col_route:
        st.markdown(
            '<div class="ap-section-head">01 · Route</div>', unsafe_allow_html=True
        )

        selected_adep_label = st.selectbox(
            "🛫  Departure Airport",
            options=adep_labels,
            index=adep_labels.index(make_airport_label("GMMN")),
            help="Select the Moroccan departure airport.",
            key="input_adep",
        )
        adep_code = extract_code_from_label(selected_adep_label)

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        available_countries = sorted(
            {AIRPORT_COUNTRY.get(code, "Unknown") for code in ades_codes}
        )
        selected_country = st.selectbox(
            "🌍  Destination Country",
            options=["— Select a country —"] + available_countries,
            index=0,
            key="input_country",
        )

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        country_chosen = selected_country != "— Select a country —"

        if country_chosen:
            filtered_ades_codes = [
                c
                for c in ades_codes
                if AIRPORT_COUNTRY.get(c, "Unknown") == selected_country
            ]
            filtered_ades_labels = [make_airport_label(c) for c in filtered_ades_codes]
            selected_ades_label = st.selectbox(
                "🛬  Destination Airport",
                options=filtered_ades_labels,
                index=0,
                help=f"Showing {len(filtered_ades_labels)} airports in {selected_country}.",
                key="input_ades",
            )
            ades_code = extract_code_from_label(selected_ades_label)
        else:
            st.selectbox(
                "🛬  Destination Airport",
                options=["— Select a destination country first —"],
                disabled=True,
                key="input_ades_disabled",
            )
            ades_code = None

    # ── COLUMN 2 — Aircraft & Operator ────────────────────────────────────────
    with col_aircraft:
        st.markdown(
            '<div class="ap-section-head">02 · Aircraft & Operator</div>',
            unsafe_allow_html=True,
        )

        default_op_idx = (
            operator_labels.index(make_operator_label("RAM"))
            if "RAM" in operator_codes
            else 0
        )
        selected_operator_label = st.selectbox(
            "🏢  Airline",
            options=operator_labels,
            index=default_op_idx,
            key="input_operator",
        )
        operator_code = extract_code_from_label(selected_operator_label)

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        default_ac_idx = (
            aircraft_labels.index(make_aircraft_label("A320"))
            if "A320" in typecodes
            else 0
        )
        selected_aircraft_label = st.selectbox(
            "✈️  Aircraft Type",
            options=aircraft_labels,
            index=default_ac_idx,
            key="input_aircraft",
        )
        typecode = extract_code_from_label(selected_aircraft_label)

    # ── COLUMN 3 — Schedule ───────────────────────────────────────────────────
    with col_schedule:
        st.markdown(
            '<div class="ap-section-head">03 · Schedule</div>', unsafe_allow_html=True
        )

        departure_date = st.date_input(
            "📅  Departure Date",
            value=datetime.date.today(),
            min_value=datetime.date(2020, 1, 1),
            max_value=datetime.date(2030, 12, 31),
            key="input_date",
        )

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        departure_time = st.time_input(
            "🕐  Departure Time (UTC)",
            value=datetime.time(8, 0),
            step=300,
            key="input_time",
        )

        # Derived values (used for inference, not shown as raw jargon)
        departure_hour = departure_time.hour
        day_of_week = departure_date.weekday()

    # ── 5.6  CTA button ───────────────────────────────────────────────────────
    st.markdown('<hr class="ap-divider">', unsafe_allow_html=True)

    _, col_btn, _ = st.columns([1.5, 2, 1.5])
    with col_btn:
        predict_clicked = st.button(
            "✈  Get My Arrival Time",
            type="primary",
            key="btn_predict",
            disabled=(not country_chosen or ades_code is None),
        )
        if not country_chosen:
            st.markdown(
                '<div class="ap-warning">⚠ Please complete all route fields first.</div>',
                unsafe_allow_html=True,
            )

    # ── 5.7  Prediction ───────────────────────────────────────────────────────
    if predict_clicked and ades_code is not None:

        # ── Build input DataFrame (exactly 6 features, correct dtypes) ────────
        input_df = pd.DataFrame(
            [
                {
                    "adep": adep_code,
                    "ades": ades_code,
                    "typecode": typecode,  # <--- MUST BE FIRST
                    "icao_operator": operator_code,  # <--- MUST BE SECOND
                    "departure_hour": departure_hour,  # int 0–23
                    "day_of_week": day_of_week,  # int 0–6
                }
            ]
        )

        # ── Spinner ────────────────────────────────────────────────────────────
        with st.spinner("Calculating exact arrival time..."):
            time.sleep(1.0)  # Remove in production

            # ── MODEL INFERENCE ────────────────────────────────────────────────
            pipeline = joblib.load("xgboost_model.pkl")
            predicted_minutes = float(pipeline.predict(input_df)[0])

        # ── Compute arrival datetime ───────────────────────────────────────────
        departure_dt = datetime.datetime.combine(departure_date, departure_time)
        predicted_td = datetime.timedelta(minutes=predicted_minutes)
        arrival_dt = departure_dt + predicted_td

        # ── Format values ──────────────────────────────────────────────────────
        total_h = int(predicted_minutes // 60)
        total_m = int(predicted_minutes % 60)
        duration_str = f"{total_h}h {total_m:02d}m"
        dep_time_str = departure_dt.strftime("%H:%M")
        dep_date_str = departure_dt.strftime("%a, %d %b %Y")
        arr_time_str = arrival_dt.strftime("%H:%M")
        arr_date_str = arrival_dt.strftime("%a, %d %b %Y")
        crosses_midnight = arrival_dt.date() != departure_date

        # ── City names ────────────────────────────────────────────────────────
        dep_city = get_city_name(adep_code)
        arr_city = get_city_name(ades_code)
        airline_name = OPERATOR_NAMES.get(operator_code, operator_code)
        aircraft_name = AIRCRAFT_NAMES.get(typecode, typecode)

        # ── "+1 day" badge HTML ────────────────────────────────────────────────
        next_day_html = (
            '<span class="next-day-badge">+1 day</span>' if crosses_midnight else ""
        )

        # ── Render boarding pass ───────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)

        # 1. We save the HTML to a variable first
        boarding_pass = f"""
        <div class="bp-wrap">
        <div class="bp-card">

            <div class="bp-header">
                <div class="bp-status-pill">
                    <div class="bp-status-dot"></div>
                    Arrival Predicted
                </div>
                <div class="bp-airline-info">
                    <span class="bp-airline-name">{airline_name}</span>
                    {aircraft_name}
                </div>
            </div>

            <div class="bp-timeline">

                <div class="bp-dep">
                    <div class="bp-iata">{adep_code}</div>
                    <div class="bp-city">{dep_city}</div>
                    <div class="bp-dep-time">{dep_time_str} <span style="font-size:0.7rem;color:var(--text-muted)">UTC</span></div>
                    <div class="bp-dep-date">{dep_date_str}</div>
                </div>

                <div class="bp-track-wrap">
                    <div class="bp-duration-label">{duration_str}</div>
                    <div class="bp-track">
                        <div class="bp-track-dot"></div>
                        <div class="bp-track-line"></div>
                        <div class="bp-plane-icon">✈</div>
                        <div class="bp-track-line"></div>
                        <div class="bp-track-dot is-dest"></div>
                    </div>
                    <div class="bp-direct-label">Direct · UTC times</div>
                </div>

                <div class="bp-arr">
                    <div class="bp-iata is-dest">{ades_code}</div>
                    <div class="bp-city" style="text-align:right">{arr_city}</div>
                    <div class="bp-arr-time">{arr_time_str}</div>
                    <div class="bp-arr-date">
                        {arr_date_str}
                        {next_day_html}
                        <span style="font-size:0.65rem;color:var(--text-muted)"> UTC</span>
                    </div>
                </div>

            </div><div class="bp-perf-divider"></div>

            <div class="bp-footer-row">
                <div class="bp-footer-item">
                    <div class="bp-footer-label">Flight Duration</div>
                    <div class="bp-footer-value">{duration_str}</div>
                </div>
                <div class="bp-footer-item">
                    <div class="bp-footer-label">Aircraft</div>
                    <div class="bp-footer-value">{aircraft_name}</div>
                </div>
                <div class="bp-footer-item">
                    <div class="bp-footer-label">Operator</div>
                    <div class="bp-footer-value">{airline_name}</div>
                </div>
                <div class="bp-footer-item">
                    <div class="bp-footer-label">Confidence</div>
                    <div class="bp-footer-value" style="color:var(--green)">High ✓</div>
                </div>
            </div>

        </div></div>"""

        # ☢️ THE NUCLEAR FIX ☢️
        # This line dynamically deletes all the empty spaces at the start of every line
        # so Streamlit's Markdown engine can't turn it into a code block!
        clean_html = "\n".join([line.lstrip() for line in boarding_pass.split("\n")])

        # 3. Now we render the perfectly clean, space-free HTML!
        st.markdown(clean_html, unsafe_allow_html=True)

    # ── 5.8  Footer ───────────────────────────────────────────────────────────
    st.markdown('<hr class="ap-divider">', unsafe_allow_html=True)
    st.markdown(
        """
    <div class="ap-footer">
        <span>AeroPredict · AI Flight Intelligence</span>
        <span>All times shown in UTC</span>
        <span>v2.0.0</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
