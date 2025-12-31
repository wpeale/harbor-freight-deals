"""Harbor Freight coupon scraper."""

import re
import time
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup


HF_COUPON_BASE_URL = "https://go.harborfreight.com"
HF_QUERY_DELAY = 0.05

@dataclass
class Coupon:
    """Represents a Harbor Freight coupon."""

    name: str
    price: float
    code: str
    expiration: str
    image_url: str
    url: str


class HarborFreightScraper:
    """Scrapes coupons from go.harborfreight.com."""

    def __init__(self):
        """Initialize scraper with configurable delay between requests.
        """
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )

    def _fetch_page(self, url: str) -> BeautifulSoup:
        """Fetch a page and return parsed BeautifulSoup."""
        response = self._session.get(url, timeout=30)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def get_total_pages(self) -> int:
        """Fetch page 1 and parse pagination to find total page count."""
        soup = self._fetch_page(f"{HF_COUPON_BASE_URL}/page/1/")

        # Look for pagination links - find the highest page number
        pagination = soup.find("div", class_="nav-links")
        if not pagination:
            pagination = soup.find("nav", class_="navigation")
        if not pagination:
            pagination = soup.find("div", class_="pagination")

        if pagination:
            page_links = pagination.find_all("a")
            max_page = 1
            for link in page_links:
                href = link.get("href", "")
                match = re.search(r"/page/(\d+)/", href)
                if match:
                    page_num = int(match.group(1))
                    max_page = max(max_page, page_num)
            return max_page

        # Fallback: check for page links anywhere
        all_links = soup.find_all("a", href=re.compile(r"/page/\d+/"))
        max_page = 1
        for link in all_links:
            match = re.search(r"/page/(\d+)/", link.get("href", ""))
            if match:
                page_num = int(match.group(1))
                max_page = max(max_page, page_num)

        return max_page if max_page > 1 else 8  # Default fallback

    def scrape_page(self, page_num: int) -> list[Coupon]:
        """Scrape a single page and return list of Coupon objects."""
        url = f"{HF_COUPON_BASE_URL}/page/{page_num}/"
        soup = self._fetch_page(url)
        coupons = []

        # Find all article elements (coupon containers)
        articles = soup.find_all("article")

        for article in articles:
            # Skip articles that don't look like coupons
            text_content = article.get_text()
            if "Code" not in text_content or "Exp" not in text_content:
                continue

            coupon = self._parse_article(article)
            if coupon:
                coupons.append(coupon)

        return coupons

    def _extract_name_and_url(self, article) -> Optional[tuple[str, str]]:
        """Extract product name and URL from article element."""
        title_elem = article.find("h2") or article.find("h3")
        if not title_elem:
            return None

        link_elem = title_elem.find("a") or article.find("a")
        if not link_elem:
            return None

        name = link_elem.get_text(strip=True)
        url = link_elem.get("href", "")

        # Make URL absolute if it's relative
        if url and not url.startswith("http"):
            url = HF_COUPON_BASE_URL + url

        if not name or len(name) < 5:
            return None

        return name, url

    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price from text content."""
        match = re.search(r"\$([0-9,]+\.?\d*)", text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass
        return None

    def _extract_code(self, text: str) -> str:
        """Extract coupon code from text content."""
        match = re.search(r"Code\s*[:#]?\s*(\d+)", text, re.IGNORECASE)
        return match.group(1) if match else ""

    def _extract_expiration(self, text: str) -> str:
        """Extract expiration date from text content."""
        match = re.search(r"Exp\.?\s*(\d{1,2}/\d{1,2}/\d{2,4})", text, re.IGNORECASE)
        return match.group(1) if match else ""

    def _extract_image_url(self, article) -> Optional[str]:
        """Extract image URL from article element."""
        img_elem = article.find("img")
        if img_elem:
            return img_elem.get("src") or img_elem.get("data-src")
        return None

    def _parse_article(self, article) -> Optional[Coupon]:
        """Parse a single article element into a Coupon."""
        result = self._extract_name_and_url(article)
        if not result:
            return None
        name, coupon_url = result

        text_content = article.get_text()
        code = self._extract_code(text_content)
        expiration = self._extract_expiration(text_content)
        price = self._extract_price(text_content)
        image_url = self._extract_image_url(article)

        if not all([code, expiration, price is not None, image_url]):
            return None

        return Coupon(
            name=name,
            price=price,
            code=code,
            expiration=expiration,
            image_url=image_url,
            url=coupon_url,
        )

    def scrape_all(self) -> list[Coupon]:
        """Scrape all pages with rate limiting."""
        total_pages = self.get_total_pages()
        all_coupons = []

        for page_num in range(1, total_pages + 1):
            coupons = self.scrape_page(page_num)
            all_coupons.extend(coupons)

            # Rate limiting between pages (not after the last page)
            if page_num < total_pages:
                time.sleep(HF_QUERY_DELAY)

        return all_coupons

    def to_llm_context(self, coupons: list[Coupon]) -> str:
        """Convert coupons to LLM-friendly text format.

        Returns markdown-formatted text optimized for LLM context.
        """
        lines = [f"# Harbor Freight Coupons ({len(coupons)} deals)\n"]

        for coupon in coupons:
            lines.append(f"## {coupon.name}")
            lines.append(f"- Price: ${coupon.price:,.2f}")
            lines.append(f"- Code: {coupon.code}")
            lines.append(f"- Expires: {coupon.expiration}")
            lines.append("")  # Blank line between coupons

        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    scraper = HarborFreightScraper()
    print(f"Detected {scraper.get_total_pages()} pages of coupons")

    print("\nScraping all coupons...")
    coupons = scraper.scrape_all()
    print(f"Found {len(coupons)} coupons")

    print("\n" + "=" * 50)
    print(scraper.to_llm_context(coupons[:5]))  # Preview first 5
