import pandas as pd
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../dataset')
    args = parser.parse_args()

    csv_file_path = os.path.join(args.data_dir, 'transcripts_with_split_outlines.csv')
    df = pd.read_csv(csv_file_path)

    
    #first outline
    df.loc[df['id'] == "CNN-42038", 'interview_goals'] = "The goal of this interview is to gather information about the situation of French journalist Michel Peyrard, who was arrested by the Taliban in Afghanistan while reporting on the war. The interview aims to understand the circumstances of his arrest, the charges against him, and the efforts being made to secure his release. The interview will also explore the concerns of the French government and the magazine \"Paris Match\" regarding Peyrard's situation and the potential implications for journalism in the region."
    df.loc[df['id'] == "CNN-42038", 'outline_statement'] = "You're about to interview Marion Mertens, the editor of the French magazine \"Paris Match\", about the situation of journalist Michel Peyrard, who was arrested by the Taliban in Afghanistan. The goals of the interview are to discuss the circumstances of Peyrard's arrest, the charges against him, and the efforts being made to secure his release. This will be a 10-question interview that will last approximately 15 minutes."
    df.loc[df['id'] == "CNN-42038", 'general_questions'] = ", ".join(["Can you describe the circumstances of Michel Peyrard's arrest and the charges against him?",
    "How is the French government responding to his situation, and what efforts are being made to secure his release?",
    "What is the current status of Peyrard's case, and are there any plans for a trial?",
    "How is the situation affecting Peyrard's colleagues and the broader journalism community?",
    "Are there any concerns about the safety of journalists in Afghanistan, and what measures can be taken to protect them?",
    "What is the role of 'Paris Match' in advocating for Peyrard's release, and what support is being provided to him and his family?"])
    
    #second outline
    df.loc[df['id'] == "CNN-73587", 'interview_goals'] = "The goal of this interview is to explore President George W. Bush's perspective on his recent five-nation tour of Africa, focusing on his views on the continent's economic and health challenges, particularly in regards to HIV/AIDS and security. The interview aims to delve into his proposed initiatives to address these issues, including the $15 billion allocation for global AIDS relief and the $100 million initiative for security in Africa. The conversation will also touch on his thoughts on the role of the United States in Africa and his vision for the future of US-Africa relations."
    df.loc[df['id'] == "CNN-73587", 'outline_statement'] = "You're about to interview President George W. Bush, who has just concluded a five-nation tour of Africa. The goals of this interview are to discuss his views on the continent's pressing issues, including HIV/AIDS and security, and his proposed initiatives to address these challenges. This will be a 15-minute interview, covering topics such as his thoughts on the role of the United States in Africa, his vision for US-Africa relations, and his plans for future engagement with the continent."
    df.loc[df['id'] == "CNN-73587", 'general_questions'] = ", ".join(["What were the most significant takeaways from your five-nation tour of Africa, and how do you believe the United States can best support the continent's development?",
    "Can you elaborate on your proposed $15 billion allocation for global AIDS relief, and how you plan to ensure that these funds are effectively utilized?",
    "How do you respond to critics who argue that the United States should be doing more to address the security challenges facing Africa, particularly in the context of terrorism and conflict?",
    "What role do you see the United States playing in promoting economic development and trade in Africa, and how do you believe this can be achieved?",
    "How do you envision the future of US-Africa relations, and what steps do you plan to take to strengthen these ties?"])
    
    #third outline
    df.loc[df['id'] == "1174", 'interview_goals'] = "The goal of this interview is to explore the life and career of singer-songwriter Janis Ian, focusing on her experiences as a teenager in the 1960s and the impact of her song \"Society's Child\" on her life and the music industry. The interview aims to delve into Ian's thoughts on the evolution of interracial relationships and the progress made in social acceptance since the 1960s."
    df.loc[df['id'] == "1174", 'outline_statement'] = "You're about to interview Janis Ian, a renowned singer-songwriter, about her life, career, and experiences as a teenager in the 1960s. The goals of the interview are to explore the impact of her song \"Society's Child\" on her life and the music industry, discuss the evolution of interracial relationships, and gain insight into her thoughts on social acceptance since the 1960s. This will be a 30-minute interview."
    df.loc[df['id'] == "1174", 'general_questions'] = ", ".join(["Can you share the story behind writing 'Society's Child' and its impact on your life?",
    "How did you experience racism and prejudice as a teenager in the 1960s, and how did it affect your music?",
    "What do you think is the most significant change in social acceptance since the 1960s, and how do you see it impacting your music and the music industry?",
    "How do you think your experiences as a teenager in the 1960s have influenced your music and artistic perspective?",
    "What message do you hope to convey through your music, and how do you see it resonating with audiences today?"])
    
    #fourth outline
    df.loc[df['id'] == "93298", 'interview_goals'] = "The goal of this interview is to explore Juan Gabriel Vasquez's thoughts on his new book, \"Lovers On All Saints' Day\", and his writing process. The interview aims to delve into the themes and inspirations behind the book, as well as his experiences as a writer and his views on the translation process. The conversation will also touch on his personal life, including his experiences living in different countries and his plans for the future."
    df.loc[df['id'] == "93298", 'outline_statement'] = "You're about to interview Juan Gabriel Vasquez, a renowned Colombian author. The goals of the interview are to discuss his new book, \"Lovers On All Saints' Day\", and his writing process, as well as his experiences as a writer and his views on translation. This will be a 20-minute interview that will cover his inspirations, personal life, and future plans."
    df.loc[df['id'] == "93298", 'general_questions'] = ", ".join(["Can you tell us more about the themes and inspirations behind 'Lovers On All Saints' Day'?",
    "How do you approach the translation process, and what role does the translator play in bringing your work to a new audience?",
    "You've lived in several countries throughout your life. How has this influenced your writing and your perspective on the world?",
    "What do you hope readers take away from your work, and what do you think sets your writing apart from others in the literary world?",
    "You've spoken about the importance of the short story as a genre. Can you elaborate on why you think it's a vital form of storytelling?",
    "What are your plans for the future, and do you have any new projects in the works?"])

    df.to_csv(os.path.join(args.data_dir, 'transcripts_with_split_outlines_fixed.csv'), index=False)
    print("Completed.")