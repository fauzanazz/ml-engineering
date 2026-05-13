from indonesian_banking_asr.synthetic.generator import (
    EntitySpec,
    TemplateSpec,
    render_template,
)


def test_render_template_replaces_slots_and_labels_entity_spans():
    template = TemplateSpec(
        template_id="check_installment_001",
        intent="check_installment",
        text="Saya mau cek cicilan {product_name} sebesar {amount}.",
        entities=(
            EntitySpec(type="BANKING_TERM", slot="cicilan", value="cicilan"),
            EntitySpec(type="PRODUCT_NAME", slot="product_name"),
            EntitySpec(type="AMOUNT", slot="amount"),
        ),
    )

    rendered = render_template(
        template,
        values={"product_name": "kartu kredit", "amount": "Rp1.250.000"},
    )

    assert rendered.text == "Saya mau cek cicilan kartu kredit sebesar Rp1.250.000."
    assert rendered.intent == "check_installment"
    assert rendered.template_id == "check_installment_001"
    assert rendered.entities == [
        {"type": "BANKING_TERM", "text": "cicilan", "start_char": 13, "end_char": 20},
        {"type": "PRODUCT_NAME", "text": "kartu kredit", "start_char": 21, "end_char": 33},
        {"type": "AMOUNT", "text": "Rp1.250.000", "start_char": 42, "end_char": 53},
    ]


def test_render_template_raises_for_missing_slot_value():
    template = TemplateSpec(
        template_id="check_balance_001",
        intent="check_balance",
        text="Cek saldo rekening {account_number}.",
        entities=(EntitySpec(type="ACCOUNT_NUMBER", slot="account_number"),),
    )

    try:
        render_template(template, values={})
    except ValueError as error:
        assert "account_number" in str(error)
    else:
        raise AssertionError("missing slot should fail")
