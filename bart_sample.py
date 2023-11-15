from transformers import BartForConditionalGeneration, BartTokenizer

def summarize_text(text):
    # モデルとトークナイザのロード
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    # テキストのトークナイズ
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    
    # 要約の生成
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, min_length=50, max_length=1000, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # サンプルのニュース記事
    news_article = """Finland's prime minister has accused Russia of helping migrants get into the country illegally, saying some have been helped by Russian border guards.

Finnish officials say migrants arrive by car before cycling across the border in small groups to claim asylum.

Finland shares a 1,340km (833-mile) border with Russia, Europe's longest.

The number of crossings remains small but has been rising this week. Border guards say they have registered around 89 crossings in two days.

That compares with 91 in the four months to 12 November.

Matti Pitkaniitty, a colonel in the Finnish border guard, told BBC News the migrants included citizens of countries including Iraq, Yemen and Syria who had arrived legally in Russia but were not authorised to enter Finland, which is an EU member state.

Traditionally, Russian guards haven't allowed people to arrive at the Finnish border without proper documents, he said. But he added Russian authorities had definitely changed their policy in recent months.

Many of the migrants are crossing into Finland by bicycle, exploiting an agreement allowing cycling across the border. Last week Finland banned crossings by bike.

Most of the activity has been seen around the border crossings at Nuijamaa and Vaalimaa, in south-eastern Finland.

Map
1px transparent line
At a news conference on Tuesday, Prime Minister Petteri Orpo accused Russian authorities of facilitating the illegal crossings.

It is clear that these people are helped and they are also being escorted or transported to the border by border guards, Mr Orpo said.

In 2021, thousands of migrants from Middle Eastern and African countries crossed into EU member states Poland and Lithuania after flying to Belarus, a close Russian ally.

The EU at the time accused Belarusian leader Alexander Lukashenko of using migration as a weapon of "hybrid warfare" to destabilise the bloc.

Mr Pitkaniitty said the small numbers seen so far remained manageable and stressed that Finnish authorities have tools available to them to react in case crossings increase.

Interior Minister Mari Rantanen said on Tuesday that her government was preparing to increase security on the border.

This way into the EU via Russia was much safer than other routes used by migrants, such as crossing the Mediterranean by sea, Mr Pitkaniitty added.

Once word gets around we may see a rapid increase in numbers. Smugglers and migrants don't know when the opportunity will end."""
    
    # 要約の取得
    summary = summarize_text(news_article)
    
    # 要約の表示
    print(summary)
