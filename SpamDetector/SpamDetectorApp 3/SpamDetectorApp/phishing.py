import streamlit as st
import re
import tldextract

# Common suspicious patterns
SUSPICIOUS_PATTERNS = [
    "bit.ly", "tinyurl.com", "goo.gl", "click", "login", "verify", "account", "free", "prize"
]

def scan_links(text):
    urls = re.findall(r'https?://\S+', text)
    link_info = []

    if urls:
        st.markdown("---")
        st.subheader("🔗 Link Scanner")
        for url in urls:
            domain = tldextract.extract(url).domain
            suspicious = any(pattern in url.lower() or pattern in domain for pattern in SUSPICIOUS_PATTERNS)

            info = {"url": url, "suspicious": suspicious}
            link_info.append(info)

            if suspicious:
                st.error(f"⚠️ Suspicious link detected: {url}")
            else:
                st.success(f"✅ Safe link: {url}")

        return True, link_info
    else:
        return False, None
