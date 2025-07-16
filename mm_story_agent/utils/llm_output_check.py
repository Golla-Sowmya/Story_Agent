def parse_list(output):
    try:
        pages = eval(output)
        return isinstance(pages, list)
    except Exception:
        # If eval fails, accept as plain text (will be handled as single page)
        return True