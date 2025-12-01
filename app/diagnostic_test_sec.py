#!/usr/bin/env python3
"""
Diagnostic test to see what's actually being extracted
"""

import asyncio
import json
from tools.sec_search import sec_search


async def diagnostic_test():
    """Run diagnostic tests with detailed output"""
    
    # ═══════════════════════════════════════════════════════════════
    # TEST 0: UNITED STATES STEEL
    # ═══════════════════════════════════════════════════════════════
    
    print("\n\n" + "=" * 80)
    print("DIAGNOSTIC TEST 5: US Steel Merger Info")
    print("=" * 80)
    
    result = await sec_search(
        company_name="United States Steel",
        ticker_symbol="X",
        form_types=["8-K", "10-K", "DEF 14A"],
        keywords=["merger", "Nippon", "acquisition", "transaction"],
        start_date="2023-01-01",
        end_date="2024-12-31",
        use_disk_cache=True
    )
    
    print(f"\nTotal filings found: {result.get('total_found', 0)}")
    print(f"Company: {result.get('company')}")
    print(f"CIK: {result.get('cik')}")
    
    if result.get("timeline"):
        for filing in result["timeline"][:3]:
            print(f"\n{'='*60}")
            print(f"Filing: {filing['date']} - {filing['form']}")
            print(f"{'='*60}")
            
            sections = filing.get("sections", {})
            print(f"Sections: {list(sections.keys())}")
            
            # Look for merger info
            if 'ma_activity' in sections:
                print("\n✅ Found M&A activity section")
                snippet = sections['ma_activity'][:500]
                print(f"Snippet: {snippet}...")
                
                
    
    # ═══════════════════════════════════════════════════════════════
    # TEST 1: TJX Pre-tax Margin (Most Important)
    # ═══════════════════════════════════════════════════════════════
    print("=" * 80)
    print("DIAGNOSTIC TEST 1: TJX Pre-tax Margin")
    print("=" * 80)
    
    result = await sec_search(
        ticker_symbol="TJX",
        form_types=["8-K"],  # Focus on 8-K (earnings releases)
        keywords=["pretax", "margin", "guidance", "plan", "fourth", "quarter"],
        start_date="2024-01-01",  # Narrow to Q4 FY2025 earnings
        end_date="2025-01-31",
        use_disk_cache=True
    )
    
    print(f"\nTotal filings found: {result.get('total_found', 0)}")
    
    if result.get("timeline"):
        for idx, filing in enumerate(result["timeline"][:5]):
            print(f"\n{'='*60}")
            print(f"Filing {idx+1}: {filing['date']} - {filing['form']}")
            print(f"URL: {filing['url']}")
            print(f"{'='*60}")
            
            metrics = filing.get("financial_metrics", {})
            sections = filing.get("sections", {})
            
            # Check ALL keys in metrics
            print(f"\n📊 Metrics found: {list(metrics.keys())}")
            
            # Check for pretax margin data
            if 'pretax_margin_data' in metrics:
                print("\n✅ FOUND pretax_margin_data:")
                print(json.dumps(metrics['pretax_margin_data'], indent=2))
            else:
                print("\n❌ NO pretax_margin_data")
                
                # Check if ANY pretax data exists
                pretax_keys = [k for k in metrics.keys() if 'pretax' in k.lower() or 'margin' in k.lower()]
                if pretax_keys:
                    print(f"   But found these related keys: {pretax_keys}")
                    for k in pretax_keys:
                        print(f"   {k}: {metrics[k]}")
            
            # Check sections
            print(f"\n📄 Sections found: {filing.get('sections_found', [])}")
            
            # Look for margin mentions in sections
            for section_name, section_text in sections.items():
                if 'margin' in section_text.lower():
                    print(f"\n   Section '{section_name}' mentions 'margin'")
                    # Show snippet
                    snippet = section_text[:500]
                    print(f"   Snippet: {snippet}...")
    
    # ═══════════════════════════════════════════════════════════════
    # TEST 2: Netflix ARPPU
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("DIAGNOSTIC TEST 2: Netflix ARPPU")
    print("=" * 80)
    
    result = await sec_search(
        ticker_symbol="NFLX",
        form_types=["10-K"],
        keywords=["revenue", "member", "ARM", "ARPPU", "average"],
        start_date="2023-01-01",
        end_date="2024-12-31",
        use_disk_cache=True
    )
    
    print(f"\nTotal filings found: {result.get('total_found', 0)}")
    
    if result.get("timeline"):
        for filing in result["timeline"]:
            year = filing["date"][:4]
            print(f"\n{'='*60}")
            print(f"Year {year}: {filing['form']}")
            print(f"{'='*60}")
            
            metrics = filing.get("financial_metrics", {})
            
            print(f"📊 Metrics found: {list(metrics.keys())}")
            
            # Check for ARPPU data
            if 'arppu_data' in metrics:
                print("\n✅ FOUND arppu_data:")
                print(json.dumps(metrics['arppu_data'], indent=2))
            else:
                print("\n❌ NO arppu_data")
                
                # Check related keys
                arppu_keys = [k for k in metrics.keys() if 'arppu' in k.lower() or 'arm' in k.lower() or 'revenue' in k.lower()]
                if arppu_keys:
                    print(f"   Related keys: {arppu_keys}")
                    for k in arppu_keys:
                        print(f"   {k}: {metrics[k]}")
            
            # Check membership data
            if 'membership_data' in metrics:
                print("\n✅ FOUND membership_data:")
                print(json.dumps(metrics['membership_data'], indent=2))
            
            # Look for ARM/ARPPU in sections
            sections = filing.get("sections", {})
            for section_name, section_text in sections.items():
                if 'arm' in section_text.lower() or 'arppu' in section_text.lower():
                    print(f"\n   Section '{section_name}' mentions ARM/ARPPU")
                    # Find the actual mentions
                    import re
                    arm_matches = re.findall(r'(?:ARM|ARPPU|average\s+revenue\s+per\s+member).{0,100}', section_text, re.I)
                    if arm_matches:
                        print(f"   Found: {arm_matches[0][:200]}")
    
    # ═══════════════════════════════════════════════════════════════
    # TEST 3: KKR Board Nominees
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("DIAGNOSTIC TEST 3: KKR Board Nominees")
    print("=" * 80)
    
    result = await sec_search(
        ticker_symbol="KKR",
        form_types=["DEF 14A"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        use_disk_cache=True
    )
    
    print(f"\nTotal filings found: {result.get('total_found', 0)}")
    
    if result.get("timeline"):
        for filing in result["timeline"][:1]:  # Just check first one
            print(f"\n{'='*60}")
            print(f"Filing: {filing['date']} - {filing['form']}")
            print(f"{'='*60}")
            
            nominees = filing.get("board_nominees", [])
            print(f"\n📋 Board nominees: {nominees}")
            
            if not nominees:
                print("\n❌ NO nominees found")
                
                # Check sections
                sections = filing.get("sections", {})
                print(f"\n📄 Sections available: {list(sections.keys())}")
                
                # Look for nominee-related text
                for section_name, section_text in sections.items():
                    if 'nomin' in section_text.lower() or 'elect' in section_text.lower():
                        print(f"\n   Section '{section_name}' mentions nominees/election")
                        # Look for name patterns
                        import re
                        name_pattern = r'([A-Z][a-z]+\s+[A-Z][a-z]+)'
                        names = re.findall(name_pattern, section_text[:2000])
                        print(f"   Potential names found: {names[:10]}")
    
    # ═══════════════════════════════════════════════════════════════
    # TEST 4: AMD Guidance
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("DIAGNOSTIC TEST 4: AMD Guidance")
    print("=" * 80)
    
    result = await sec_search(
        ticker_symbol="AMD",
        form_types=["8-K"],
        keywords=["guidance", "outlook", "expects", "revenue"],
        start_date="2024-10-01",
        end_date="2024-12-31",
        use_disk_cache=True
    )
    
    print(f"\nTotal filings found: {result.get('total_found', 0)}")
    
    if result.get("timeline"):
        for filing in result["timeline"][:2]:
            print(f"\n{'='*60}")
            print(f"Filing: {filing['date']} - {filing['form']}")
            print(f"{'='*60}")
            
            guidance = filing.get("guidance_data", {})
            
            if guidance:
                print("\n✅ FOUND guidance_data:")
                print(json.dumps(guidance, indent=2))
            else:
                print("\n❌ NO guidance_data")
                
                # Check sections
                sections = filing.get("sections", {})
                guidance_sections = [k for k in sections.keys() if 'guidance' in k.lower() or 'outlook' in k.lower()]
                if guidance_sections:
                    print(f"   Found guidance-related sections: {guidance_sections}")
                    for section in guidance_sections:
                        snippet = sections[section][:500]
                        print(f"\n   {section}:")
                        print(f"   {snippet}...")


if __name__ == "__main__":
    asyncio.run(diagnostic_test())
