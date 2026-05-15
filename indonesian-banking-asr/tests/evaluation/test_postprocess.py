from indonesian_banking_asr.evaluation.postprocess import postprocess_transcript


def test_postprocess_transcript_collapses_spaced_and_hyphenated_digits():
    assert postprocess_transcript("nomor 0 4 3 3 2 1 8 1 9 6") == "nomor 0433218196"
    assert postprocess_transcript("nomor 0433-218-196") == "nomor 0433218196"
    assert postprocess_transcript("nomor 0 433 218 196") == "nomor 0433218196"
    assert postprocess_transcript("akhirannya 9.027") == "akhirannya 9027"


def test_postprocess_transcript_corrects_banking_lexicon():
    assert postprocess_transcript("saldo BIFAS dan Kris aku bloket") == "saldo BI-FAST dan QRIS aku blocked"
    assert postprocess_transcript("BI Fast Beifast bevas pelater Sopiko Sopi") == "BI-FAST BI-FAST BI-FAST paylater Shopee Shopee"
    assert postprocess_transcript("Krisaya dan Chris81514235177847") == "QRIS saya dan QRIS 81514235177847"


def test_postprocess_transcript_normalizes_rupiah_amounts():
    assert postprocess_transcript("sebesar 17.110.000 rupiah") == "sebesar Rp17.110.000"
    assert postprocess_transcript("sebesar RP36010000") == "sebesar Rp36.010.000"
    assert postprocess_transcript("sebesar Rp 17.110.000") == "sebesar Rp17.110.000"
