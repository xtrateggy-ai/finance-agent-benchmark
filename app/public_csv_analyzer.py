#!/usr/bin/env python3
"""
Public.csv Question Pattern Analyzer
Identifies all unique question types and required data extraction patterns
"""

import pandas as pd
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import json


class QuestionPatternAnalyzer:
    """Analyze public.csv to identify question patterns and extraction requirements"""
    
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.lower()
        
        # Pattern categories
        self.patterns = {
            'merger_acquisition': [],
            'guidance': [],
            'comparison': [],
            'trend_analysis': [],
            'metric_calculation': [],
            'board_governance': [],
            'margin_analysis': [],
            'subscriber_metrics': [],
            'earnings': [],
            'valuation': [],
            'other': []
        }
        
        # Required data types
        self.data_requirements = defaultdict(set)
        
        # Filing types needed
        self.filing_requirements = defaultdict(set)
    
    def analyze_all(self) -> Dict:
        """Run complete analysis"""
        print("=" * 80)
        print("PUBLIC.CSV QUESTION PATTERN ANALYSIS")
        print("=" * 80)
        print(f"Total questions: {len(self.df)}")
        print()
        
        # Categorize questions
        self._categorize_questions()
        
        # Extract patterns
        results = {
            'total_questions': len(self.df),
            'patterns': self._analyze_patterns(),
            'keywords': self._extract_keywords(),
            'time_periods': self._analyze_time_periods(),
            'companies': self._analyze_companies(),
            'answer_types': self._analyze_answer_types(),
            'data_requirements': self._generate_requirements(),
            'extraction_priority': self._prioritize_extractions()
        }
        
        return results
    
    def _categorize_questions(self):
        """Categorize each question by pattern type"""
        for idx, row in self.df.iterrows():
            question = str(row.get('question', '')).lower()
            answer = str(row.get('answer', '')).lower()
            
            # Pattern detection
            if any(kw in question for kw in ['merger', 'acquisition', 'deal', 'bought', 'acquired']):
                self.patterns['merger_acquisition'].append((idx, question, answer))
                self.filing_requirements['merger_acquisition'].update(['8-K', '10-K', 'DEF 14A'])
            
            elif any(kw in question for kw in ['guidance', 'outlook', 'forecast', 'expects', 'projected']):
                self.patterns['guidance'].append((idx, question, answer))
                self.filing_requirements['guidance'].update(['8-K'])
                self.data_requirements['guidance'].add('forward_guidance_ranges')
            
            elif any(kw in question for kw in ['compare', 'versus', 'vs', 'difference between']):
                self.patterns['comparison'].append((idx, question, answer))
                self.data_requirements['comparison'].add('multi_company_data')
            
            elif any(kw in question for kw in ['trend', 'changed', 'over time', 'from', 'to', '2019', '2024']):
                self.patterns['trend_analysis'].append((idx, question, answer))
                self.data_requirements['trend_analysis'].add('time_series_data')
            
            elif any(kw in question for kw in ['arppu', 'arpu', 'per user', 'per member', 'per subscriber']):
                self.patterns['subscriber_metrics'].append((idx, question, answer))
                self.data_requirements['subscriber_metrics'].update(['revenue', 'subscriber_count', 'arppu_calculation'])
                self.filing_requirements['subscriber_metrics'].update(['10-K', '10-Q'])
            
            elif any(kw in question for kw in ['board', 'director', 'nominee', 'elected']):
                self.patterns['board_governance'].append((idx, question, answer))
                self.filing_requirements['board_governance'].update(['DEF 14A', '8-K'])
                self.data_requirements['board_governance'].add('board_nominee_names')
            
            elif any(kw in question for kw in ['margin', 'profitability', 'pre-tax']):
                self.patterns['margin_analysis'].append((idx, question, answer))
                self.data_requirements['margin_analysis'].update(['gross_margin', 'operating_margin', 'pretax_margin'])
            
            elif any(kw in question for kw in ['earnings', 'eps', 'profit', 'income']):
                self.patterns['earnings'].append((idx, question, answer))
                self.data_requirements['earnings'].update(['net_income', 'eps', 'diluted_shares'])
            
            else:
                self.patterns['other'].append((idx, question, answer))
    
    def _analyze_patterns(self) -> Dict:
        """Analyze distribution of question patterns"""
        distribution = {}
        
        print("\nQUESTION PATTERN DISTRIBUTION:")
        print("-" * 80)
        
        for pattern_name, questions in self.patterns.items():
            count = len(questions)
            pct = (count / len(self.df)) * 100 if len(self.df) > 0 else 0
            
            distribution[pattern_name] = {
                'count': count,
                'percentage': round(pct, 1),
                'examples': [q[1][:100] for q in questions[:3]]
            }
            
            if count > 0:
                print(f"{pattern_name:.<30} {count:>4} ({pct:>5.1f}%)")
        
        return distribution
    
    def _extract_keywords(self) -> Dict:
        """Extract most common keywords by category"""
        keywords_by_category = {}
        
        print("\n\nKEY TERMS BY CATEGORY:")
        print("-" * 80)
        
        # Financial metrics
        financial_terms = Counter()
        for _, row in self.df.iterrows():
            question = str(row.get('question', '')).lower()
            for term in ['revenue', 'income', 'assets', 'liabilities', 'equity', 
                        'cash flow', 'margin', 'profit', 'eps', 'ebitda']:
                if term in question:
                    financial_terms[term] += 1
        
        print("\nFinancial Metrics:")
        for term, count in financial_terms.most_common(10):
            print(f"  {term:.<30} {count:>4}")
        
        keywords_by_category['financial_metrics'] = dict(financial_terms.most_common(20))
        
        # Time periods
        time_terms = Counter()
        for _, row in self.df.iterrows():
            question = str(row.get('question', '')).lower()
            # Extract years
            years = re.findall(r'\b(20\d{2})\b', question)
            for year in years:
                time_terms[year] += 1
            # Extract quarters
            quarters = re.findall(r'\b(Q[1-4])\b', question, re.I)
            for q in quarters:
                time_terms[q.upper()] += 1
        
        print("\nTime Periods:")
        for term, count in time_terms.most_common(15):
            print(f"  {term:.<30} {count:>4}")
        
        keywords_by_category['time_periods'] = dict(time_terms.most_common(20))
        
        return keywords_by_category
    
    def _analyze_time_periods(self) -> Dict:
        """Analyze time period patterns in questions"""
        time_analysis = {
            'year_ranges': [],
            'specific_years': Counter(),
            'quarters': Counter(),
            'fiscal_years': []
        }
        
        for _, row in self.df.iterrows():
            question = str(row.get('question', '')).lower()
            
            # Year ranges (2019-2024, 2019 to 2024)
            year_ranges = re.findall(r'(20\d{2})(?:\s*(?:to|-)\s*)(20\d{2})', question)
            for start, end in year_ranges:
                time_analysis['year_ranges'].append(f"{start}-{end}")
            
            # Specific years
            years = re.findall(r'\b(20\d{2})\b', question)
            for year in years:
                time_analysis['specific_years'][year] += 1
            
            # Quarters
            quarters = re.findall(r'\b(Q[1-4])\b', question, re.I)
            for q in quarters:
                time_analysis['quarters'][q.upper()] += 1
            
            # Fiscal year mentions
            if 'fiscal' in question or 'fy' in question:
                time_analysis['fiscal_years'].append(question[:100])
        
        print("\n\nTIME PERIOD ANALYSIS:")
        print("-" * 80)
        print(f"Year ranges found: {len(time_analysis['year_ranges'])}")
        if time_analysis['year_ranges']:
            print(f"  Examples: {', '.join(time_analysis['year_ranges'][:5])}")
        
        print(f"\nMost common years:")
        for year, count in time_analysis['specific_years'].most_common(10):
            print(f"  {year}: {count}")
        
        print(f"\nQuarters mentioned:")
        for q, count in time_analysis['quarters'].most_common():
            print(f"  {q}: {count}")
        
        return time_analysis
    
    def _analyze_companies(self) -> Dict:
        """Analyze company mentions"""
        companies = Counter()
        
        # Common company patterns
        company_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Capitalized names
        ]
        
        for _, row in self.df.iterrows():
            question = str(row.get('question', ''))
            
            # Known companies (expand this list)
            known_companies = [
                'Netflix', 'Apple', 'Microsoft', 'Amazon', 'Google', 'Meta',
                'Tesla', 'NVIDIA', 'AMD', 'Intel', 'Cisco', 'Oracle',
                'US Steel', 'Nippon Steel', 'TJX', 'BBSI'
            ]
            
            for company in known_companies:
                if company.lower() in question.lower():
                    companies[company] += 1
        
        print("\n\nMOST MENTIONED COMPANIES:")
        print("-" * 80)
        for company, count in companies.most_common(20):
            print(f"  {company:.<30} {count:>4}")
        
        return dict(companies.most_common(50))
    
    def _analyze_answer_types(self) -> Dict:
        """Analyze answer format patterns"""
        answer_types = {
            'numeric': 0,
            'percentage': 0,
            'currency': 0,
            'text': 0,
            'list': 0,
            'yes_no': 0,
            'range': 0
        }
        
        for _, row in self.df.iterrows():
            answer = str(row.get('answer', ''))
            
            if re.search(r'\$[\d,]+(?:\.\d+)?', answer):
                answer_types['currency'] += 1
            elif re.search(r'\d+(?:\.\d+)?%', answer):
                answer_types['percentage'] += 1
            elif re.search(r'\d+(?:\.\d+)?(?:\s*(?:billion|million|thousand))?', answer):
                answer_types['numeric'] += 1
            elif ' to ' in answer.lower() or '-' in answer:
                answer_types['range'] += 1
            elif ',' in answer and len(answer.split(',')) > 2:
                answer_types['list'] += 1
            elif answer.lower() in ['yes', 'no', 'true', 'false']:
                answer_types['yes_no'] += 1
            else:
                answer_types['text'] += 1
        
        print("\n\nANSWER FORMAT DISTRIBUTION:")
        print("-" * 80)
        for answer_type, count in answer_types.items():
            pct = (count / len(self.df)) * 100 if len(self.df) > 0 else 0
            print(f"  {answer_type:.<30} {count:>4} ({pct:>5.1f}%)")
        
        return answer_types
    
    def _generate_requirements(self) -> Dict:
        """Generate data extraction requirements by pattern"""
        requirements = {}
        
        print("\n\nDATA EXTRACTION REQUIREMENTS:")
        print("=" * 80)
        
        for pattern_name, data_needed in self.data_requirements.items():
            filings_needed = self.filing_requirements.get(pattern_name, set())
            
            requirements[pattern_name] = {
                'data_types': list(data_needed),
                'filing_types': list(filings_needed),
                'question_count': len(self.patterns.get(pattern_name, []))
            }
            
            if requirements[pattern_name]['question_count'] > 0:
                print(f"\n{pattern_name.upper()}")
                print(f"  Questions: {requirements[pattern_name]['question_count']}")
                print(f"  Data needed: {', '.join(data_needed)}")
                print(f"  Filings: {', '.join(filings_needed)}")
        
        return requirements
    
    def _prioritize_extractions(self) -> List[Dict]:
        """Prioritize extraction implementations by impact"""
        priority_list = []
        
        for pattern_name, questions in self.patterns.items():
            if len(questions) == 0:
                continue
            
            data_needed = self.data_requirements.get(pattern_name, set())
            filings_needed = self.filing_requirements.get(pattern_name, set())
            
            # Calculate priority score
            score = len(questions) * 10  # Weight by question count
            
            if not data_needed:
                score *= 0.5  # Lower priority if requirements unclear
            
            priority_list.append({
                'pattern': pattern_name,
                'priority_score': score,
                'question_count': len(questions),
                'data_requirements': list(data_needed),
                'filing_types': list(filings_needed),
                'example_questions': [q[1][:80] for q in questions[:2]]
            })
        
        priority_list.sort(key=lambda x: x['priority_score'], reverse=True)
        
        print("\n\nIMPLEMENTATION PRIORITY:")
        print("=" * 80)
        print(f"{'Rank':<6} {'Pattern':<25} {'Questions':<12} {'Priority Score':<15}")
        print("-" * 80)
        
        for rank, item in enumerate(priority_list, 1):
            print(f"{rank:<6} {item['pattern']:<25} {item['question_count']:<12} {item['priority_score']:<15.0f}")
        
        return priority_list
    
    def generate_report(self, output_file: str = "question_analysis_report.json"):
        """Generate comprehensive JSON report"""
        results = self.analyze_all()
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n\n{'='*80}")
        print(f"REPORT SAVED: {output_file}")
        print(f"{'='*80}")
        
        return results
    
    def generate_extraction_guide(self):
        """Generate practical extraction implementation guide"""
        print("\n\n" + "=" * 80)
        print("PRACTICAL EXTRACTION GUIDE")
        print("=" * 80)
        
        guide = {
            'merger_acquisition': {
                'files': ['8-K Item 1.01', '8-K Item 2.01', '10-K Risk Factors'],
                'patterns': [
                    r'merger|acquisition|deal',
                    r'agreed to (?:acquire|purchase)',
                    r'transaction valued at'
                ],
                'extract': ['deal_terms', 'transaction_value', 'closing_date']
            },
            'guidance': {
                'files': ['8-K Item 2.02', 'Earnings Releases'],
                'patterns': [
                    r'guidance|outlook|expects?',
                    r'\$[\d.]+.*billion.*to.*\$[\d.]+.*billion',
                    r'margin.*\d+\.\d+%.*to.*\d+\.\d+%'
                ],
                'extract': ['revenue_range', 'margin_range', 'eps_guidance']
            },
            'subscriber_metrics': {
                'files': ['10-K Item 8 Notes', '10-Q Revenue Recognition'],
                'patterns': [
                    r'paid.*members?hip',
                    r'average.*revenue.*per.*user',
                    r'subscribers?'
                ],
                'extract': ['member_count', 'revenue_per_user', 'arppu_calculation']
            },
            'board_governance': {
                'files': ['DEF 14A', '8-K Item 5.02'],
                'patterns': [
                    r'(?:nominat|elect).*director',
                    r'Board.*(?:nominates|elects)',
                    r'([A-Z][a-z]+\s+[A-Z][a-z]+).*director'
                ],
                'extract': ['nominee_names', 'election_date', 'board_class']
            }
        }
        
        for pattern, details in guide.items():
            print(f"\n{pattern.upper()}")
            print(f"  Files to check: {', '.join(details['files'])}")
            print(f"  Regex patterns:")
            for p in details['patterns']:
                print(f"    - {p}")
            print(f"  Data to extract: {', '.join(details['extract'])}")


def main():
    """Run analysis on public.csv"""
    csv_path = "data/public.csv"
    
    # Check if file exists
    import os
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found!")
        print("Please place public.csv in the data/ directory")
        return
    
    # Run analysis
    analyzer = QuestionPatternAnalyzer(csv_path)
    results = analyzer.generate_report()
    analyzer.generate_extraction_guide()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total questions analyzed: {results['total_questions']}")
    print(f"Unique patterns identified: {len([p for p, d in results['patterns'].items() if d['count'] > 0])}")
    print(f"Report saved: question_analysis_report.json")
    print("\nNext steps:")
    print("1. Review priority rankings to guide implementation")
    print("2. Use extraction guide to update sec_search.py")
    print("3. Test extraction patterns against sample filings")


if __name__ == "__main__":
    main()
