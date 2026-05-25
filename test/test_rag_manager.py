from rag.manager import RagManager


rag = RagManager()

doc_id = rag.add_text(
    "秋天到了，校园里的银杏树渐渐变黄。每天早晨，我走进学校的时候，"
    "都会看到地上铺满金黄色的落叶，像一张柔软的地毯。微风吹过，"
    "树叶轻轻飘落，同学们一边走路一边欢笑，整个校园显得格外安静又美丽。"
    "操场上，同学们正在认真上体育课，有的人在跑步，有的人在打篮球，"
    "充满了活力。教室里，老师正在给我们讲课，窗外偶尔传来鸟叫声，"
    "让人感到十分舒服。我喜欢秋天，因为它不像夏天那样炎热，"
    "也不像冬天那样寒冷。秋天不仅有美丽的景色，还有丰收的气息。"
    "农田里的稻谷成熟了，果园里的水果也变得香甜可口。"
    "我觉得，秋天是一个温暖而又充满希望的季节。",
    metadata={"source": "test_rag_manager"},
)

print("doc_id:", doc_id)
print("stats:", rag.get_stats())

print("search results:")
for chunk in rag.search("同学们在做什么", limit=3):
    print(f"- score={chunk.score:.4f}: {chunk.content}")

print("context:")
print(rag.build_context("同学们在做什么", limit=3))
