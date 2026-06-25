# Expected security mapping: stock symbol (data.xls Sheet1 header) -> fundamentals
# company label (data.xls Sheet2 "Company name").
#
# This dict is NOT the source of truth. The authoritative mapping is derived at
# runtime by joining the two sheets on ISIN (the unambiguous security identifier)
# in `data_loader.derive_code_to_name()`. This table only documents the expected
# result; `derive_code_to_name` raises if the file ever disagrees with it, so the
# original silent mislabelling (stock returns paired with the wrong company's
# fundamentals) cannot recur.
#
# Verified against data.xls by ISIN:
#   B01NPJ  INE467B01029 -> Tata Consultancy Services
#   620512  INE009A01021 -> Infosys
#   629489  INE860A01027 -> HCL Technologies
#   620605  INE075A01022 -> Wipro
#   BWFGD6  INE669C01036 -> Tech Mahindra
EXPECTED_STOCK_CODE_TO_NAME = {
    'B01NPJ': 'Tata Consultancy Services Ltd.',
    '620512': 'Infosys Ltd.',
    '629489': 'HCL Technologies Ltd.',
    '620605': 'Wipro Ltd.',
    'BWFGD6': 'Tech Mahindra Ltd.',
}
