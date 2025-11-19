from livekit.agents.voice.filler_detection.semantic_detector import SemanticFillerDetector

if __name__ == '__main__':
    s = SemanticFillerDetector()
    if s.model is None:
        print('Semantic model not available, check sentence-transformers installation')
    else:
        print('Semantic model loaded:', s.model_name)
        tests = ['hmmmm', 'ummmmm', 'stop', 'wait', 'you know']
        for t in tests:
            r = s.calculate_similarity(t)
            print(t, '-> filler:', r['is_filler'], 'filler_sim', round(r['filler_similarity'], 3), 'cmd_sim', round(r['command_similarity'],3))
