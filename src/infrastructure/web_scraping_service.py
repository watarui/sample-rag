"""Webスクレイピングサービス実装"""

from __future__ import annotations

import asyncio
import re
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from ..domain.entities import Document
from ..domain.services import WebScrapingService
from ..domain.value_objects import (
    DocumentContent,
    DocumentId,
    ScrapingConfig,
    Timestamp,
    WebPageContent,
    WebSource,
)
from ..shared.result import Result, try_catch_async


class WebScrapingServiceImpl(WebScrapingService):
    """Webスクレイピングサービス実装"""
    
    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        self._user_agent = UserAgent()
        self._robots_cache: dict[str, RobotFileParser] = {}
        print("Initialized Web scraping service")
    
    async def _get_session(self, config: ScrapingConfig) -> aiohttp.ClientSession:
        """HTTPセッションを取得"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
            headers = {
                'User-Agent': config.user_agent or self._user_agent.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                connector=aiohttp.TCPConnector(limit=10)
            )
        return self._session
    
    async def _check_robots_txt(self, url: str, config: ScrapingConfig) -> bool:
        """robots.txtをチェック"""
        # 一時的にrobots.txtチェックを無効化
        print(f"Skipping robots.txt check for {url}")
        return True
    
    async def scrape_single_page(
        self,
        url: str,
        config: ScrapingConfig
    ) -> Result[Document, Exception]:
        """単一のWebページをスクレイピング"""
        @try_catch_async
        async def _scrape() -> Document:
            # robots.txtチェック
            if not await self._check_robots_txt(url, config):
                raise Exception(f"robots.txt disallows scraping {url}")
            
            session = await self._get_session(config)
            
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status} for {url}")
                
                html_content = await response.text()
                content_type = response.headers.get('content-type', 'text/html')
            
            # BeautifulSoupでパース
            soup = BeautifulSoup(html_content, 'lxml')
            
            # タイトルを抽出
            title = self._extract_title(soup, config)
            
            # メインコンテンツを抽出
            text_content = self._extract_content(soup, config)
            
            # リンクを抽出
            links = self._extract_links(soup, url, config)
            
            # メタデータを抽出
            metadata = self._extract_metadata(soup, url)
            
            # WebPageContentを作成
            page_content = WebPageContent(
                url=url,
                title=title,
                text_content=text_content,
                html_content=html_content,
                metadata=metadata,
                links=links,
            )
            
            # WebSourceを作成
            parsed_url = urlparse(url)
            web_source = WebSource(
                url=url,
                title=title,
                domain=parsed_url.netloc,
                scraped_at=Timestamp(),
                content_type=content_type,
            )
            
            # DocumentContentを作成
            document_content = DocumentContent(
                text=page_content.get_clean_text(),
                metadata={
                    "title": title,
                    "url": url,
                    "domain": parsed_url.netloc,
                    "scraped_at": str(Timestamp()),
                    "content_type": content_type,
                    "source": f"Web({parsed_url.netloc})",
                    **metadata,
                },
            )
            
            # Documentを作成
            document = Document(
                id=DocumentId(),
                title=title or f"Page from {parsed_url.netloc}",
                content=document_content,
                source=web_source,
                tags=["web", "scraped", parsed_url.netloc],
            )
            
            print(f"Successfully scraped: {url}")
            return document
        
        return await _scrape()
    
    def _extract_title(self, soup: BeautifulSoup, config: ScrapingConfig) -> str:
        """タイトルを抽出"""
        for selector in config.title_selectors:
            elements = soup.select(selector)
            if elements:
                title = elements[0].get_text().strip()
                if title:
                    return title
        
        # フォールバック
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        return ""
    
    def _extract_content(self, soup: BeautifulSoup, config: ScrapingConfig) -> str:
        """メインコンテンツを抽出"""
        # 除外要素を削除
        for selector in config.exclude_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # メインコンテンツを抽出
        content_parts = []
        
        for selector in config.content_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(separator=' ', strip=True)
                if text and len(text) > 50:  # 最小長チェック
                    content_parts.append(text)
        
        # メインコンテンツが見つからない場合はbody全体を使用
        if not content_parts:
            body = soup.find('body')
            if body:
                content_parts.append(body.get_text(separator=' ', strip=True))
        
        content = '\n\n'.join(content_parts)
        
        # テキストクリーニング
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str, config: ScrapingConfig) -> List[str]:
        """リンクを抽出"""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            
            # 同じドメインまたは許可されたドメインのみ
            parsed_url = urlparse(absolute_url)
            if (not config.allowed_domains or 
                parsed_url.netloc in config.allowed_domains or
                parsed_url.netloc == urlparse(base_url).netloc):
                links.append(absolute_url)
        
        return list(set(links))  # 重複除去
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> dict:
        """メタデータを抽出"""
        metadata = {}
        
        # メタタグ
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        
        # 見出し
        headings = []
        for h_tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = h_tag.get_text().strip()
            if text:
                headings.append(text)
        
        if headings:
            metadata['headings'] = headings[:10]  # 最大10個
        
        return metadata
    
    async def discover_pages(
        self,
        base_url: str,
        config: ScrapingConfig
    ) -> Result[List[str], Exception]:
        """ベースURLから関連ページを発見"""
        @try_catch_async
        async def _discover() -> List[str]:
            discovered_urls: Set[str] = set()
            to_visit = [base_url]
            visited = set()
            
            for depth in range(config.max_depth + 1):
                if not to_visit or len(discovered_urls) >= config.max_pages:
                    break
                
                current_level = to_visit.copy()
                to_visit.clear()
                
                for url in current_level:
                    if url in visited or len(discovered_urls) >= config.max_pages:
                        continue
                    
                    visited.add(url)
                    
                    try:
                        # レート制限
                        await asyncio.sleep(config.delay_seconds)
                        
                        # ページをスクレイピング
                        result = await self.scrape_single_page(url, config)
                        if result.is_success():
                            document = result.unwrap()
                            discovered_urls.add(url)
                            
                            # リンクを次のレベルに追加
                            if hasattr(document.source, 'links'):
                                for link in document.source.links:
                                    if link not in visited and len(discovered_urls) < config.max_pages:
                                        to_visit.append(link)
                    
                    except Exception as e:
                        print(f"Error discovering page {url}: {e}")
                        continue
            
            return list(discovered_urls)
        
        return await _discover()
    
    async def scrape_pages(
        self,
        config: ScrapingConfig,
        query: str = "",
        limit: int = 10
    ) -> Result[List[Document], Exception]:
        """Webページをスクレイピングして文書を生成"""
        @try_catch_async
        async def _scrape_pages() -> List[Document]:
            print(f"Starting web scraping with query: '{query}' (limit: {limit})")
            
            # 直接base_urlsをスクレイピング
            urls_to_scrape = config.base_urls[:limit]
            
            if not urls_to_scrape:
                print("No URLs configured for scraping")
                return []
            
            print(f"Scraping {len(urls_to_scrape)} URLs directly")
            
            # 並列でページをスクレイピング
            tasks = []
            for url in urls_to_scrape:
                task = asyncio.create_task(self.scrape_single_page(url, config))
                tasks.append(task)
            
            # 全てのタスクを実行
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            documents = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Error scraping page {i}: {result}")
                    continue
                
                if hasattr(result, 'is_success') and result.is_success():
                    document = result.unwrap()
                    
                    # クエリフィルタリング（オプション）
                    if query and query.lower() not in document.content.text.lower():
                        print(f"Skipping document due to query filter: {document.title}")
                        continue
                    
                    documents.append(document)
                else:
                    print(f"Failed to scrape page {i}: {result}")
            
            print(f"Successfully scraped {len(documents)} documents")
            return documents
        
        return await _scrape_pages()
    
    async def close(self) -> None:
        """リソースをクリーンアップ"""
        if self._session and not self._session.closed:
            await self._session.close() 