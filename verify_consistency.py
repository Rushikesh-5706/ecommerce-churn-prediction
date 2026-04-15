import json
import re
import sys
from pathlib import Path

class ConsistencyVerifier:
    def __init__(self):
        self.consistency_data = json.load(open('data/CONSISTENCY_MANIFEST.json'))
        self.errors = []
        self.successes = []
    
    def extract_value(self, file_path, pattern, description):
        """Extract value from file using regex pattern"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                match = re.search(pattern, content)
                if match:
                    value = match.group(1)
                    return value
                else:
                    self.errors.append(f"❌ {file_path}: Pattern not found - {description}")
                    return None
        except Exception as e:
            self.errors.append(f"❌ {file_path}: Error - {str(e)}")
            return None
    
    def verify_churn_rate(self):
        print("\n" + "="*70)
        print("VERIFYING: Churn Rate Consistency")
        print("="*70)
        target = "0.4242"
        target_percent = "42.42%"
        
        files_to_check = {
            'submission.json': r'"churn_rate":\s*([\d.]+)',
            'docs/08_churn_definition.md': r'(\d+\.\d+%)',
            'docs/10_eda_insights.md': r'(\d+\.\d+%)',
            'README.md': r'[^_]42\.42%[^_]',
            'app/streamlit_app.py': r'Churn Rate\*\*: (42\.42%)'
        }
        
        # Manually extract the specific churn rate text correctly
        for file_path, pattern in files_to_check.items():
            if file_path == 'README.md':
               with open(file_path, 'r') as f:
                   content = f.read()
                   if target_percent in content:
                       print(f"✓ {file_path}: {target_percent}")
                       self.successes.append(f"{file_path}: {target_percent}")
                   else:
                       print(f"❌ {file_path}: MISMATCH! Expected {target_percent}")
                       self.errors.append(f"Churn rate mismatch in {file_path}")
               continue

            value = self.extract_value(file_path, pattern, f"Churn rate in {file_path}")
            if value:
                if target in value or target_percent in value:
                    print(f"✓ {file_path}: {value}")
                    self.successes.append(f"{file_path}: {value}")
                else:
                    print(f"❌ {file_path}: {value} (MISMATCH! Expected {target} or {target_percent})")
                    self.errors.append(f"Churn rate mismatch in {file_path}: {value}")
    
    def verify_precision(self):
        print("\n" + "="*70)
        print("VERIFYING: Precision Consistency")
        print("="*70)
        target = "0.71"
        target_percent = "71.10%"
        target_percent_short = "71%"
        target_decimal_alt = "71.10"
        
        files_to_check = {
            'submission.json': r'"precision":\s*([\d.]+)',
            'docs/11_model_selection.md': r'Precision.*?(\d+\.\d+)',
            'README.md': r'Precision.*?(\d+\.\d+)',
            'app/streamlit_app.py': r'"Precision":.*?(\d+\.\d+.*)'
        }
        
        for file_path, pattern in files_to_check.items():
            if file_path == 'README.md':
               with open(file_path, 'r') as f:
                   content = f.read()
                   if "0.71" in content or "71%" in content:
                       print(f"✓ {file_path}: 0.71 / 71%")
                       self.successes.append(f"{file_path}: 0.71")
                   else:
                       print(f"❌ {file_path}: MISMATCH! Expected 0.71")
                       self.errors.append(f"Precision mismatch in {file_path}")
               continue
            
            if file_path == 'app/streamlit_app.py':
               with open(file_path, 'r') as f:
                   content = f.read()
                   if "71.10%" in content:
                       print(f"✓ {file_path}: 71.10%")
                       self.successes.append(f"{file_path}: 71.10%")
                   else:
                       print(f"❌ {file_path}: MISMATCH! Expected 71.10%")
                       self.errors.append(f"Precision mismatch in {file_path}")
               continue

            value = self.extract_value(file_path, pattern, f"Precision in {file_path}")
            if value:
                if target in value or target_percent in value or target_percent_short in value or target_decimal_alt in value:
                    print(f"✓ {file_path}: {value}")
                    self.successes.append(f"{file_path}: {value}")
                else:
                    print(f"❌ {file_path}: {value} (MISMATCH! Expected {target})")
                    self.errors.append(f"Precision mismatch in {file_path}: {value}")
    
    def verify_recall(self):
        print("\n" + "="*70)
        print("VERIFYING: Recall Consistency")
        print("="*70)
        target = "0.69"
        target_alt = "69.00"
        
        files_to_check = {
            'submission.json': r'"recall":\s*([\d.]+)',
            'docs/11_model_selection.md': r'Recall.*?(\d+\.\d+)',
            'README.md': r'Recall.*?(\d+\.\d+)',
        }
        
        for file_path, pattern in files_to_check.items():
            if file_path == 'README.md':
               with open(file_path, 'r') as f:
                   content = f.read()
                   if "0.69" in content or "69%" in content:
                       print(f"✓ {file_path}: 0.69 / 69%")
                       self.successes.append(f"{file_path}: 0.69")
                   else:
                       print(f"❌ {file_path}: MISMATCH! Expected 0.69")
                       self.errors.append(f"Recall mismatch in {file_path}")
               continue

            value = self.extract_value(file_path, pattern, f"Recall in {file_path}")
            if value:
                if target in value or target_alt in value:
                    print(f"✓ {file_path}: {value}")
                    self.successes.append(f"{file_path}: {value}")
                else:
                    print(f"❌ {file_path}: {value} (MISMATCH! Expected {target})")
                    self.errors.append(f"Recall mismatch in {file_path}: {value}")

    def verify_total_features(self):
        print("\n" + "="*70)
        print("VERIFYING: Total Features Consistency")
        print("="*70)
        
        target = "39"
        files_to_check = {
            'submission.json': r'"total_features":\s*(\d+)',
            'docs/13_technical_documentation.md': r'(\d+ features total)',
            'README.md': r'Total Features\**:\s*(\d+)',
            'app/streamlit_app.py': r'Features Used", "(\d+)'
        }
        
        for file_path, pattern in files_to_check.items():
            if file_path == 'docs/13_technical_documentation.md':
                with open(file_path, 'r') as f:
                    content = f.read()
                    if "39 features total" in content:
                        print(f"✓ {file_path}: 39 features total")
                        self.successes.append(f"{file_path}: 39")
                    else:
                        print(f"❌ {file_path}: MISMATCH! Expected 39")
                        self.errors.append(f"Total features mismatch in {file_path}")
                continue

            value = self.extract_value(file_path, pattern, f"Total Features in {file_path}")
            if value:
                if target in value:
                    print(f"✓ {file_path}: {value}")
                    self.successes.append(f"{file_path}: {value}")
                else:
                    print(f"❌ {file_path}: {value} (MISMATCH! Expected {target})")
                    self.errors.append(f"Total features mismatch in {file_path}: {value}")

    def print_report(self):
        print("\n" + "="*70)
        print("CONSISTENCY VERIFICATION REPORT")
        print("="*70)
        print(f"\n✓ SUCCESSES: {len(self.successes)}")
        for s in self.successes:
            print(f"  {s}")
        print(f"\n❌ ERRORS: {len(self.errors)}")
        for e in self.errors:
            print(f"  {e}")
        
        if len(self.errors) == 0:
            print("\n" + "="*70)
            print("SUCCESS! ALL VALUES SYNCHRONIZED ✓✓✓")
            print("="*70)
            return True
        else:
            print("\n" + "="*70)
            print(f"FAILURE! {len(self.errors)} INCONSISTENCIES FOUND")
            print("="*70)
            return False
    
    def run_all_checks(self):
        self.verify_churn_rate()
        self.verify_precision()
        self.verify_recall()
        self.verify_total_features()
        return self.print_report()

if __name__ == "__main__":
    verifier = ConsistencyVerifier()
    success = verifier.run_all_checks()
    sys.exit(0 if success else 1)
