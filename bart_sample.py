from transformers import BartForConditionalGeneration, BartTokenizer

def summarize_text(text):
    # モデルとトークナイザのロード
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    # テキストのトークナイズ
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    
    # 要約の生成
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, min_length=400, max_length=3000, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # サンプルのニュース記事
    news_article = """千葉市中心部にイノシシ 警察官など3人けが
2023年10月25日 12時36分 

25日未明、千葉市の中心部にイノシシが現れ、目撃した男性1人がけがをしました。イノシシは午前9時すぎに捕獲されましたが、この際、捕獲にあたった警察官2人もかまれてけがをしました。

千葉市中央区の住宅街などで、25日午前0時すぎから1時すぎにかけて、イノシシの目撃情報が相次いで寄せられました。

警察や消防によりますと、このうち、30代の男性が千葉市中央区出洲港の住宅の敷地内で左足をかまれ、病院に搬送されました。

目撃情報はほかにも、千葉駅前のロータリーなど千葉市中央区の中心部で合わせて13件あったということです。

その後、午前9時ごろ、中央区中央港のふ頭で警察官らがイノシシを捕獲しましたが、この際、警察官2人が足をかまれ、けがをしました。

捕獲される前には海を泳ぐなどして逃げ回っていたということです。

千葉市によりますと、イノシシは体重が42キロ余り、体長が1メートル余りのメスで、捕獲後、山林に埋めて処分されたということです。

イノシシの出没を受けて、千葉市内では警察が住宅街や学校の付近を中心にパトロールを行い、注意を呼びかけました。

北海道 札幌市内でもシカの目撃情報相次ぐ 
一方、札幌市では25日、シカの目撃情報が相次いで寄せられ、警察などが付近で警戒にあたっています。

北海道大学植物園は24日、シカが迷い込んだため臨時休園しましたが、25日はいないことが確認され、開園しています。

札幌市東区東苗穂にある住宅の庭で25日、角のあるシカ1頭が確認されました。

シカは木陰に寝そべったり、時折、庭の中を移動して草のようなものを食べたりしていて、警察や区の職員が警戒にあたっています。

住民は午前8時ごろに、警察からシカがいることを知らされたということで、「ここに住んで50年以上になるがシカが出たのは初めてで、びっくりしている。角があるので怖い」と話していました。

また、警察によりますと、これに先立ち、25日午前1時半ごろ、札幌市中央区北4条西1丁目のあたりで、午前2時10分ごろには東区北7条東3丁目のあたりで、それぞれ、通行人からシカを目撃したとの通報があったということです。

一方、中央区にある北海道大学植物園は24日、角のあるシカ1頭が迷い込んだため臨時休園しましたが、25日朝、園内にいないことが確認されたとして、通常より1時間遅い午前10時から開園しています。

札幌市では、これまでに目撃されたシカと、植物園にいたシカの関連も含めて調べています。"""
    
    # 要約の取得
    summary = summarize_text(news_article)
    
    # 要約の表示
    print(summary)
