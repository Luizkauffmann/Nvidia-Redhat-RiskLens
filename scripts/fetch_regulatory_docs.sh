#!/usr/bin/env bash
# Download public-domain / redistributable CECL & IFRS 9 regulatory documents
# into docs/regulatory/sources/. Run from the repo root on a machine with open
# internet access:   bash scripts/fetch_regulatory_docs.sh
# Copyrighted standards (FASB ASC, IFRS) are linked in docs/regulatory/README.md only.
set -u
OUT="$(cd "$(dirname "$0")/.." && pwd)/docs/regulatory/sources"
mkdir -p "$OUT"
UA="Mozilla/5.0 (RiskLens regulatory fetch)"

docs=(
  "us_interagency_policy_statement_ACL_rev2023.pdf|https://public-inspection.federalregister.gov/2023-08876.pdf"
  "us_interagency_ACL_and_credit_risk_review_2020_release.pdf|https://archive.fdic.gov/view/fdic/15345/fdic_15345_DS1.pdf"
  "us_capital_rule_CECL_transition_2019.pdf|https://www.govinfo.gov/content/pkg/FR-2019-02-14/pdf/2018-28281.pdf"
  "us_capital_rule_CECL_revised_transition_2020.pdf|https://www.govinfo.gov/content/pkg/FR-2020-09-30/pdf/2020-19782.pdf"
  "us_SR26-2_model_risk_guidance_2026_FDIC.pdf|https://www.fdic.gov/model-risk-management-revised-guidance.pdf"
  "us_SR11-7_model_risk_2011_superseded.pdf|https://www.federalreserve.gov/supervisionreg/srletters/sr1107a1.pdf"
  "us_SEC_SAB119.pdf|https://www.sec.gov/files/rules/staff/2019/sab-119.pdf"
  "intl_BCBS_d350_credit_risk_and_ECL_accounting_2015.pdf|https://www.bis.org/bcbs/publ/d350.pdf"
  "eu_ECB_IFRS9_overlays_novel_risks_2024.pdf|https://www.bankingsupervision.europa.eu/ecb/pub/pdf/ssm.IFRS9novelrisks_202407~5e0eb30b5c.en.pdf"
  "eu_ESRB_wp92_cyclicality_SICR_IFRS9.pdf|https://www.esrb.europa.eu/pub/pdf/wp/esrb.wp92~bf769a68b8.en.pdf"
)

ok=0; fail=0
for entry in "${docs[@]}"; do
  name="${entry%%|*}"; url="${entry#*|}"
  if curl -fsSL -A "$UA" --max-time 90 -o "$OUT/$name.part" "$url" \
     && head -c 5 "$OUT/$name.part" | grep -q '%PDF'; then
    mv "$OUT/$name.part" "$OUT/$name"; echo "OK    $name"; ok=$((ok+1))
  else
    rm -f "$OUT/$name.part"; echo "FAIL  $name  <- $url (open it in a browser and save manually)"; fail=$((fail+1))
  fi
done
echo "Done: $ok downloaded, $fail failed -> $OUT"
