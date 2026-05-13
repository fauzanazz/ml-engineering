from pathlib import Path

from indonesian_banking_asr.synthetic.catalog import load_template_catalog


def test_load_template_catalog_reads_yaml_templates(tmp_path):
    catalog_path = tmp_path / "templates.yaml"
    catalog_path.write_text(
        """
- template_id: check_balance_001
  intent: check_balance
  text: "Saya mau cek saldo {product_name} nomor {account_number}."
  entities:
    - type: PRODUCT_NAME
      slot: product_name
    - type: ACCOUNT_NUMBER
      slot: account_number
""".strip()
    )

    templates = load_template_catalog(catalog_path)

    assert len(templates) == 1
    assert templates[0].template_id == "check_balance_001"
    assert templates[0].intent == "check_balance"
    assert templates[0].entities[0].type == "PRODUCT_NAME"
    assert templates[0].entities[1].slot == "account_number"


def test_default_catalog_has_small_intent_coverage():
    templates = load_template_catalog(Path("data/templates/banking_intents.yaml"))
    intents = {template.intent for template in templates}

    assert len(templates) >= 12
    assert intents == {
        "check_balance",
        "check_transaction_history",
        "check_installment",
        "credit_card_limit",
        "loan_interest_rate",
        "mortgage_interest_rate",
        "account_blocked",
        "card_blocked",
        "transfer_failed",
        "virtual_account_payment",
        "change_phone_number",
        "complaint_fee_charge",
    }
