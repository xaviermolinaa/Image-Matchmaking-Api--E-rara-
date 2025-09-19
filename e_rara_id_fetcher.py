import requests
import re
import time
from bs4 import BeautifulSoup
from urllib.parse import urlencode

# e-rara SRU endpoint and IIIF image base
SRU_URL = "https://www.e-rara.ch/search"

def build_cql_query(author=None, from_date=None, until_date=None, place=None, title=None, publisher=None):
    """
    Build a CQL query for searching e-rara's SRU interface with date handling.
    
    Parameters:
    -----------
    author : str, optional
        Author name to search for
    from_date : str or int, optional
        Start date for search range (e.g., "1800" or 1800)
    until_date : str or int, optional
        End date for search range (e.g., "1850" or 1850)
    place : str, optional
        Publication place to search for
    title : str, optional
        Title to search for
    publisher : str, optional
        Publisher name to search for
    
    Returns:
    --------
    str
        The formatted CQL query string
    """
    if author and isinstance(author, str):
        author = author.strip()
    if title and isinstance(title, str):
        title = title.strip()
    if publisher and isinstance(publisher, str):
        publisher = publisher.strip()
    if place and isinstance(place, str):
        place = place.strip()
    if publisher and isinstance(publisher, str):
        publisher = publisher.strip()

    cql_parts = []
    if author:
        cql_parts.append(f'dc.creator all "{author}"')
    if title:
        cql_parts.append(f'dc.title all "{title}"')
    if publisher:
        cql_parts.append(f'vl.printer-publisher all "{publisher}"')
    if place:
        cql_parts.append(f'bib.originPlace all "{place}"')
    if from_date or until_date:
        from_ = str(from_date) if from_date is not None else "*"
        until = str(until_date) if until_date is not None else "*"
        cql_parts.append(f'dc.date={from_}-{until}')
    
    return " AND ".join(cql_parts) if cql_parts else None


def _parse_xml_response(content, start_record, max_records_page):
    """
    Parse SRU XML response with handling of different XML structures.
    """
    soup = BeautifulSoup(content, 'xml')

    total_elem = soup.find('numberOfRecords')
    total = int(total_elem.text) if total_elem else 0
    print(f"Found total records: {total}")
    
    ids = []
    
    records = soup.find_all('record')
    print(f"Found {len(records)} record elements")
    
    # Method 1: Look for recordIdentifier tags directly
    record_identifiers = soup.find_all('recordIdentifier')
    for rec_id in record_identifiers:
        if rec_id.text and rec_id.text.startswith('oai:www.e-rara.ch:'):
            ids.append(rec_id.text.split(':')[-1])
    
    if ids:
        print(f"Found {len(ids)} IDs using recordIdentifier tags")
        next_start = (start_record + max_records_page) if (start_record + max_records_page) <= total else None
        return ids, next_start, total
    
    # Method 2: Look for dc:identifier elements in each record
    for record in records:
        identifier_elements = []
        
        record_data = record.find('recordData')
        if record_data:
            identifier_elements.extend(record_data.find_all('identifier'))
            
            identifier_elements.extend(record_data.find_all('dc:identifier'))
            
            for elem in record_data.find_all():
                if 'identifier' in elem.name.lower():
                    identifier_elements.append(elem)
        
        for identifier in identifier_elements:
            if identifier.text and 'oai:www.e-rara.ch:' in identifier.text:
                record_id = identifier.text.split(':')[-1]
                if record_id not in ids:
                    ids.append(record_id)
            
            elif identifier.text and 'e-rara.ch' in identifier.text and '/viewer/' in identifier.text:
                match = re.search(r'/viewer/(\d+)', identifier.text)
                if match:
                    record_id = match.group(1)
                    if record_id not in ids:
                        ids.append(record_id)
            
            elif identifier.text and identifier.text.isdigit():
                if identifier.text not in ids:
                    ids.append(identifier.text)
    
    # Method 3: As a last resort, look for any numeric IDs in the response
    if not ids:
        print("No IDs found using standard methods, searching for numeric patterns in the XML...")
        for tag in soup.find_all():
            if tag.text and re.match(r'^\d{7,9}$', tag.text.strip()):
                if tag.text.strip() not in ids:
                    ids.append(tag.text.strip())
    
    print(f"Extracted {len(ids)} record IDs using all methods")
    
    next_start = (start_record + max_records_page) if (start_record + max_records_page) <= total else None
    
    return ids, next_start, total

def _parse_html_response(html_content, start_record, max_records_page=100):
    """
    HTML response parser that extracts record IDs from e-rara's HTML responses.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    total = 0
    span = soup.select_one('span.titlecount')
    if span:
        try:
            total = int(span.text.strip().replace(',', ''))
        except ValueError:
            pass

    ids = []
    
    # Method 1: Look for content/titleinfo links
    # print("Num of <a> tags in HTML:", len(soup.find_all('a', href=True)))
    # print("Num of <img> tags in HTML:", len(soup.find_all('img', src=True)))
    # print("Num of <span> tags in HTML:", len(soup.find_all('span')))
    for a in soup.find_all('a', href=True):
        href = a['href']
        # Pattern: "/content/titleinfo/12345678"
        if '/titleinfo/' in href:
            identifier = href.split('/')[-1]
            try:
                identifier = str(int(identifier))
                if identifier not in ids:
                    ids.append(identifier)
            except ValueError:
                continue
            
    # Method 2: Look for viewer links if no titleinfo links found
    if not ids:
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Pattern: "/viewer?identifier=12345678"
            if '/viewer/' in href and 'identifier=' in href:
                identifier = href.split('identifier=')[-1]
                if identifier.isdigit() and identifier not in ids:
                    ids.append(identifier)
    
    # Method 3: Try to find IDs in image source URLs if no IDs found yet
    if not ids:
        for img in soup.find_all('img', src=True):
            src = img['src']
            # Pattern: "/i3f/v21/12345678/full/"
            if '/i3f/v21/' in src and '/full/' in src:
                parts = src.split('/')
                for i, part in enumerate(parts):
                    if part == 'v21' and i+1 < len(parts) and parts[i+1].isdigit():
                        identifier = parts[i+1]
                        if identifier not in ids:
                            ids.append(identifier)
    
    print(f"Extracted {len(ids)} record IDs from HTML response")
    
    next_start = start_record + max_records_page if total and (start_record + max_records_page) <= total else None

    return ids, next_start, total


def sru_search(cql_query, start_record=1, max_records_page=100, truncation="on"):
    """
    Search e-rara using SRU protocol and extract record IDs.
    
    This function handles both XML and HTML responses from the SRU service
    and falls back to different parsing methods if needed.
    
    Parameters:
    -----------
    cql_query : str
        The CQL query string
    start_record : int, optional
        Starting record number (for pagination)
    max_records : int, optional
        Maximum number of records to return
    truncation : str, optional
        Whether to enable truncation ("on" or "off")
        
    Returns:
    --------
    tuple
        (record_ids, next_start, total_records) where:
        - record_ids is a list of e-rara IDs
        - next_start is the next starting position (or None)
        - total_records is the total number of matches
    """
    params = {
        'operation': 'searchRetrieve', 
        'version': '2.1',
        'query': cql_query, 
        'startRecord': start_record,
        'maximumRecords': max_records_page, 
        'recordSchema': 'dc',
        'truncate': truncation,
    }

    url = SRU_URL + '?' + urlencode(params)
    print(f"Request URL: {url}")

    resp = requests.get(SRU_URL, params=params)
    resp.raise_for_status()
    
    print(f"Response status: {resp.status_code}")
    print(f"Response content type: {resp.headers.get('content-type', 'Unknown')}")
    
    content = resp.content.lstrip()
    if content.startswith(b'<!DOCTYPE html>') or b'<html' in content.lower():
        print("Response contains HTML, using HTML parser")
        return _parse_html_response(resp.text, start_record, max_records_page)
    else:
        print("Response appears to be XML, using XML parser")
    
    try:
        return _parse_xml_response(resp.content, start_record, max_records_page)
    except Exception as e:
        print(f"XML parsing failed: {e}")
        print("Falling back to HTML parsing...")
        return _parse_html_response(resp.text, start_record, max_records_page)
    

def search_ids(cql_query, start_record=1, max_records=None, max_records_page=100, truncation="on"):
    """
    Search for IDs using a CQL query and handle pagination.
    
    Parameters:
    -----------
    cql_query : str
        The CQL query string
    start_record : int, optional
        Starting record number (for pagination)
    max_records : int, optional
        Maximum number of records to return
    truncation : str, optional
        Whether to enable truncation ("on" or "off")

    Returns:
    --------
    tuple
        (record_ids, total_records) where:
        - record_ids is a list of e-rara IDs
        - total_records is the total number of matches
    """
    all_ids = []
    total = 0
    next_start = start_record
    while next_start:
        print(f"Fetching records {next_start}-{next_start+max_records_page-1}...")
        ids, next_start, total = sru_search(cql_query, next_start, max_records_page, truncation=truncation)
        
        if not ids:
            print("No more records found, stopping pagination")
            break
            
        all_ids.extend(ids)
        print(f"Total IDs collected so far: {len(all_ids)}")
        
        all_ids = list(set(all_ids))
        if max_records and len(all_ids) >= max_records:
            print(f"Reached {max_records} limit of {max_records}. Stopping search.")
            break
        
        time.sleep(1)

    if max_records and len(all_ids) > max_records:
        all_ids = all_ids[:max_records]
        print(f"Returning first {max_records} records out of {len(all_ids)} collected.")
    
    print(f"Final collection: {len(all_ids)} IDs out of {total} total records")
    return all_ids, total


def build_cql_query_v2(place=None, title=None, author=None, publisher=None, start_record=1,
                             from_date=None, until_date=None, truncation='on', domain="erara"):
    """
    Build a query that matches the format used by the e-rara website's search form.
    
    Parameters:
    -----------
    place : str, optional
        Publication place to search for
    title : str, optional
        Title to search for
    author : str, optional
        Author/creator name to search for
    publisher : str, optional
        Publisher/printer name to search for
    from_date : str, optional
        Start date for search range (e.g., "1800")
    until_date : str, optional
        End date for search range (e.g., "1850")
    domain : str, optional
        Domain to search within (default: 'erara')
        
    Returns:
    --------
    str
        The formatted query string
    dict
        Parameters to use in the request
    """
    if place and isinstance(place, str):
        place = place.strip()
    if title and isinstance(title, str):
        title = title.strip()
    if author and isinstance(author, str):
        author = author.strip()
    if publisher and isinstance(publisher, str):
        publisher = publisher.strip()
    if start_record and isinstance(start_record, int):
        start_record = max(1, start_record)
    
    query_parts = []
    if place:
        query_parts.append(f"bib.originPlace={place}")
    if title:
        query_parts.append(f"dc.title={title}")
    if author:
        query_parts.append(f"dc.creator={author}")
    if publisher:
        query_parts.append(f"vl.printer-publisher={publisher}")
    if domain:
        query_parts.append(f"vl.domain=({domain})")

    
    if from_date or until_date:
        from_ = from_date if from_date else "*"
        until = until_date if until_date else "*"
        query_parts.append(f"dc.date={from_}-{until}")
    
    query = " and ".join([f"({part})" for part in query_parts])
    if query:
        query += " sortBy relevance/desc"
    
    params = {
        'operation': 'searchRetrieve',
        'query': query,
        'index1': 'bib.originPlace',
        'term1': place if place else '',
        'bool2': 'and',
        'index2': 'dc.title',
        'term2': title if title else '',
        'bool3': 'and',
        'index3': 'dc.creator',
        'term3': author if author else '',
        'bool4': 'and',
        'index4': 'vl.printer-publisher',
        'term4': publisher if publisher else '',
        'startRecord': start_record,
        'truncate': truncation
    }
    
    if from_date:
        params['from'] = from_date
    if until_date:
        params['until'] = until_date
    
    return query, params


def search_ids_v2(place=None, title=None, author=None, publisher=None, 
                from_date=None, until_date=None, start_record=1,
                max_records=None, max_record_page=100, truncation='on', use_website_style=True):
    """
    Search for ids from e-rara based on various filters using the website-style approach.
    This matches the behavior and results seen on the e-rara website.
    
    Parameters:
    -----------
    place : str, optional
        Publication place to search for
    title : str, optional
        Title to search for
    author : str, optional
        Author/creator name to search for
    publisher : str, optional
        Publisher/printer name to search for
    from_date : str, optional
        Start date for search range (e.g., "1800")
    until_date : str, optional
        End date for search range (e.g., "1850")
    max_records : int, optional
        Maximum number of results to return
    use_website_style : bool, optional
        Whether to use the website-style search approach (default: True)
        
    Returns:
    --------
    list
        List of e-rara record IDs
    """

    if from_date and until_date and int(until_date) - int(from_date) > 399:
        print(f"WARNING: The year gap between from_date ({from_date}) and until_date ({until_date}) is more than 400 years. Splitting search into multiple requests.")
        all_ids = []
        for year in range(int(from_date), int(until_date) + 1, 400):
            next_from_date = str(year)
            next_until_date = str(min(year + 399, int(until_date)))
            print(f"Searching from {next_from_date} to {next_until_date}...")
            ids = search_ids_v2(
                place=place,
                title=title,
                author=author,
                publisher=publisher,
                from_date=next_from_date,
                until_date=next_until_date,
                start_record=start_record,
                max_records=max_records,
                max_record_page=max_record_page,
                truncation=truncation,
                use_website_style=use_website_style
            )
            all_ids.extend(ids)

        all_ids = list(set(all_ids))
        print(f"Total unique records found across all requests: {len(all_ids)}")
        return all_ids


    if use_website_style:
        
        if place: print(f"- Place: {place}")
        if title: print(f"- Title: {title}")
        if author: print(f"- Author: {author}")
        if publisher: print(f"- Publisher: {publisher}")
        if from_date: print(f"- From date: {from_date}")
        if until_date: print(f"- Until date: {until_date}")
        if start_record!=1: print(f"- Start record: {start_record}")
        if max_records: print(f"- Max records: {max_records}")
        if truncation: print(f"- Truncation: {truncation}")
        if not max_records:
            print("WARNING: No max_records specified. Fatching all foud ids.")
        if not start_record:
            start_record = 1
        
        if max_record_page not in [10, 20, 30, 50, 100]:
            print(f"WARNING: max_record_page should be one of [10, 20, 30, 50, 100]. Using default value of 100.")
            max_record_page = 100
        
        _, params = build_cql_query_v2(
            place=place,
            title=title,
            author=author,
            publisher=publisher,
            from_date=from_date,
            until_date=until_date,
            start_record=start_record,
            truncation=truncation
        )
        
        all_ids = []
        params['maximumRecords'] = max_record_page
        next_start = start_record
        while next_start:
            print(f"Fetching records starting from {next_start}...")
            params['startRecord'] = next_start
            url = SRU_URL + '?' + urlencode(params)
            print(f"Request URL: {url}")
            resp = requests.get(SRU_URL, params=params)
            resp.raise_for_status()
        
            ids, next_start, total = _parse_html_response(resp.text, next_start, max_record_page)
            all_ids.extend(ids)
            all_ids = list(set(all_ids))
            
            time.sleep(1)

            if max_records and len(all_ids) >= max_records:
                print(f"Reached max_records limit of {max_records}. Stopping search.")
                break
        
        print(f"Found {total} total records, {len(all_ids)} unique records.")
        if max_records and len(all_ids) > max_records:
            all_ids = all_ids[:max_records]
            print(f"Returning first {max_records} records.")
        return all_ids, total

    else:
        cql_query = build_cql_query(
            author=author, 
            from_date=from_date, 
            until_date=until_date,
            place=place, 
            title=title,
            publisher=publisher
        )
        
        if not cql_query:
            print("WARNING: No filters specified. Please set at least one filter.")
            return []
        
        print(f"CQL query: {cql_query}")
        
        record_ids, total = search_ids(cql_query,
                                       start_record=start_record,
                                       max_records=max_records,
                                       truncation=truncation)
        print(f"Found {total} total records, fetched {len(record_ids)}")
        return record_ids, total


if __name__ == "__main__":

    # Example usage: change these as needed
    filters = dict(
        # author=None,            # e.g. "Goethe"
        from_date="1600",         # e.g. "1800"
        until_date="1620",        # e.g. "1850"
        place="Bern",           # e.g. "Bern"
        # title="Radio",             # e.g. "Faust"
        # publisher=None,         # e.g. "Schiller"
        # max_records=923,          # e.g. 5
        use_website_style=True,  # e.g. True
        truncation="on"         # e.g. "on"/"off"
    )

    print("Testing enhanced search with Place='Bern' and max_records=5:")
    results_bern, total = search_ids_v2(**filters)
    # print(f"Results for 'Bern': {results_bern}")

    with open('ids.txt', 'w') as f:
        for record_id in results_bern:
            f.write(f"{record_id}\n")

    # Also test with CQL approach for comparison
    # filters['use_website_style'] = False
    # print("\nTesting CQL search with Place='Bern' and max_records=5:")
    # results_cql_bern = search_ids_v2(**filters)
    # print(f"Results for CQL 'Bern': {results_cql_bern}")

    # print("\nTesting enhanced search with Title='Historia' and max_records=5:")
    # filters['place'] = None
    # filters['title'] = "Historia"
    # filters['use_website_style'] = True
    # results_historia = search_ids_v2(**filters)
    # print(f"Results for 'Historia': {results_historia}")

    # print("\nTesting CQL search with Title='Historia' and max_records=5:")
    # filters['use_website_style'] = False
    # results_cql = search_ids_v2(**filters)
    # print(f"Results for CQL 'Historia': {results_cql}")
