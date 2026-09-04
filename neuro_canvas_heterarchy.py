import time
import math
import numpy as np
import torch
import torch.nn as nn
import pygame

# ==============================================================================
# КОНФИГУРАЦИЯ И АППАРАТНЫЕ КОНСТАНТЫ
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WIDTH, HEIGHT = 1200, 800

# Координаты 16 пинов датчика FreeEEG16-alpha2 (26 mm)
COORDS_X = np.array([ 10.14,  7.43,  2.75,  2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72,  2.72,  2.75,  7.43,  10.14])
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42,  -2.73,   2.72,  7.43,  4.76,  10.14, 10.15,  4.77,  7.42,   2.71])

# ==============================================================================
# 1. КАНОНИЧЕСКАЯ КОЛОНКА ХОКИНСА (TBT SPATIAL POOLER + L2/3 + L5a)
# ==============================================================================
class CanonicalHTMColumn(nn.Module):
    """
    Реализует паттерны `tbp.monty` (Thousand Brains Theory) & Working Memory 2.0.
    Принимает честный 2D тензор [120 ребер, 32 гамма-такта].
    """
    def __init__(self, name: str, num_columns: int = 2304, k_active: int = 46):
        super().__init__()
        self.name = name
        self.num_columns = num_columns # 48x48 2D сетка
        self.k_active = k_active       # 2% разреженность

        # 1. Создаем топологическую матрицу перманентности (L4)
        # Инициализируем синапсы равномерно вокруг порога 0.5
        init_perm = torch.empty(num_columns, 120, device=DEVICE).uniform_(0.35, 0.65)
        self.register_buffer("permanence", init_perm)
        self.perm_threshold = 0.50

        # 2. Ассоциативная память (Kanerva SDM) для Слоя 5a (Генеративный выход)
        # Хранит координаты (g, s)
        self.register_buffer("W_assoc", torch.zeros((2, num_columns), device=DEVICE))
        self.register_buffer("col_counts", torch.zeros(num_columns, device=DEVICE))

        # 3. Гомеостатический Бустинг
        self.register_buffer("duty_cycle", torch.full((num_columns,), 0.02, device=DEVICE))
        self.last_sdr = torch.zeros(num_columns, device=DEVICE)
        self.anomaly_score = 0.0

    def forward(self, pac_iplv_120x32: torch.Tensor, is_learning: bool = False):
        """
        pac_iplv_120x32: [120, 32] - Направленный граф iPLV на 32 гамма-слотах
        """
        # A. Временное мультиплексирование (Working Memory 2.0)
        # Придаем больший вес последним гамма-слотам (проспекция цели)
        phase_weights = torch.linspace(0.5, 1.5, 32, device=DEVICE).unsqueeze(0)
        integrated_spatial = torch.sum(pac_iplv_120x32 * phase_weights, dim=1) / 32.0 # [120]

        # B. Вычисление Overlap (L4)
        connected_synapses = (self.permanence >= self.perm_threshold).float()
        overlap = torch.mv(connected_synapses, integrated_spatial) # [2304]

        # C. Гомеостатический бустинг (СТРОГО ТОЛЬКО ПРИ ОБУЧЕНИИ)
        if is_learning:
            target_duty = self.k_active / self.num_columns
            boost_factors = torch.exp(-3.0 * (self.duty_cycle - target_duty))
            overlap = overlap * boost_factors

        # D. k-WTA Ингибирование (Формирование SDR)
        _, active_indices = torch.topk(overlap, self.k_active)
        sdr = torch.zeros(self.num_columns, device=DEVICE)
        sdr[active_indices] = 1.0

        # E. Вычисление Аномалии Хокинса (L2/3 mismatch)
        if self.last_sdr.sum() > 0:
            intersection = (sdr * self.last_sdr).sum()
            self.anomaly_score = 1.0 - (intersection / self.k_active).item()
        self.last_sdr = sdr.clone()

        # F. Хеббовское обучение синапсов
        if is_learning:
            # Обновляем Duty Cycle (скользящее среднее)
            self.duty_cycle = self.duty_cycle * 0.99 + sdr * 0.01

            # Укрепляем синапсы, если вход был активен, ослабляем иначе
            delta_p = torch.where(integrated_spatial > 0.5, 0.08, -0.02).unsqueeze(0)
            self.permanence[active_indices, :] = torch.clamp(
                self.permanence[active_indices, :] + delta_p, 0.0, 1.0
            )

        return sdr, active_indices

    def imprint_target(self, active_indices: torch.Tensor, target_val: torch.Tensor):
        """ Запоминание цели в слое L5a (без перезаписи, защита от забывания) """
        self.W_assoc[:, active_indices] += target_val.unsqueeze(1)
        self.col_counts[active_indices] += 1.0

    def recall_target(self, active_indices: torch.Tensor):
        """ Извлечение генеративной команды L5a (Консенсусное голосование) """
        votes = self.W_assoc[:, active_indices]
        counts = self.col_counts[active_indices].clamp(min=1.0)
        consensus = torch.sum(votes / counts, dim=1) / float(self.k_active)
        return torch.clamp(consensus, 0.0, 1.0)


# ==============================================================================
# 2. ПРЕФРОНТАЛЬНАЯ ГЕТЕРАРХИЯ (F3, F4, AFz, Fpz)
# ==============================================================================
class PrefrontalHeterarchy(nn.Module):
    def __init__(self):
        super().__init__()
        self.f3  = CanonicalHTMColumn("F3_Form")     # Декодирует ось G
        self.f4  = CanonicalHTMColumn("F4_Style")    # Декодирует ось S
        self.afz = CanonicalHTMColumn("AFz_Torus")   # Шлюзование
        self.fpz = CanonicalHTMColumn("Fpz_Branch")  # План Б / Phase Reset

        self.plan_b_active = False

    def learn_calibration(self, tensors_120x32, target_g, target_s, plan_b_g, plan_b_s):
        # Прогоняем колонки в режиме обучения (is_learning=True - бустинг включен!)
        _, act_f3  = self.f3(tensors_120x32['F3'], is_learning=True)
        _, act_f4  = self.f4(tensors_120x32['F4'], is_learning=True)
        _, act_fpz = self.fpz(tensors_120x32['Fpz'], is_learning=True)

        target_f3 = torch.tensor([target_g, 0.0], device=DEVICE)
        target_f4 = torch.tensor([0.0, target_s], device=DEVICE)
        target_fpz = torch.tensor([plan_b_g, plan_b_s], device=DEVICE)

        self.f3.imprint_target(act_f3, target_f3)
        self.f4.imprint_target(act_f4, target_f4)
        self.fpz.imprint_target(act_fpz, target_fpz)

    def predict(self, tensors_120x32):
        # Инференс (is_learning=False - бустинг заморожен, ретенция сохраняется!)
        _, act_f3  = self.f3(tensors_120x32['F3'], is_learning=False)
        _, act_f4  = self.f4(tensors_120x32['F4'], is_learning=False)
        _, act_fpz = self.fpz(tensors_120x32['Fpz'], is_learning=False)

        # Логика когнитивного ветвления (Cognitive Branching по Koechlin, 2007)
        branch_anomaly = self.fpz.anomaly_score
        self.plan_b_active = branch_anomaly > 0.65

        if self.plan_b_active and self.fpz.col_counts.max() > 0:
            # Phase Reset -> Переключаемся на План Б
            pred_gs = self.fpz.recall_target(act_fpz)
            return pred_gs[0].item(), pred_gs[1].item(), True
        else:
            # План А -> F3 дает G, F4 дает S
            pred_g = self.f3.recall_target(act_f3)[0].item()
            pred_s = self.f4.recall_target(act_f4)[1].item()
            return pred_g, pred_s, False


# ==============================================================================
# 3. АВТОНОМНЫЙ АГЕНТ ЗАМКНУТОГО ЦИКЛА (Синтезирует iPLV по целям)
# ==============================================================================
class ClosedLoopAgent:
    def __init__(self):
        self.targets = [
            {"name": "ГОРА",       "g": 0.0, "s": 0.0},
            {"name": "ЗАМОК",      "g": 1.0, "s": 0.0},
            {"name": "НЕБОСКРЕБ", "g": 1.0, "s": 1.0},
            {"name": "ОКЕАН",      "g": 0.0, "s": 1.0}
        ]
        self.tgt_idx = 0
        self.satisfaction = 0.0
        self.boredom = 0.0
        self.frustration = 0.0
        self.mode = "SEEKING"
        self.t_last = time.time()

    def update_and_generate_tensors(self, screen_g, screen_s):
        dt = time.time() - self.t_last
        self.t_last = time.time()

        cur_tgt = self.targets[self.tgt_idx]
        dist = math.hypot(screen_g - cur_tgt["g"], screen_s - cur_tgt["s"])

        # Каузальный цикл Active Inference
        if dist < 0.25:
            self.satisfaction = min(1.0, self.satisfaction + dt * 0.5)
            self.frustration = max(0.0, self.frustration - dt)
            if self.satisfaction > 0.8:
                self.boredom += dt * 0.5
                self.mode = "SATISFIED"
        else:
            self.satisfaction = max(0.0, self.satisfaction - dt * 0.5)
            self.frustration += dt * 0.3
            self.boredom = max(0.0, self.boredom - dt)
            self.mode = "SEEKING"

        # Саккады (Переключение внимания)
        if self.boredom > 1.0 or self.frustration > 1.5:
            self.tgt_idx = (self.tgt_idx + 1) % 4
            self.satisfaction = 0.0
            self.boredom = 0.0
            self.frustration = 0.0
            self.mode = "SACCADE"

        # Генерация "мыслей" агента: уникальные [120, 32] паттерны для каждой цели
        # Имитация ортогональных сигналов фазовой синхронизации (iPLV)
        tensors_120x32 = {}
        for region in ['F3', 'F4', 'AFz', 'Fpz']:
            np.random.seed(hash(region + cur_tgt['name']) % (2**32))
            base_pattern = np.random.uniform(0, 1, (120, 32))
            noise = np.random.normal(0, 0.1, (120, 32))
            t = torch.tensor(np.clip(base_pattern + noise, 0, 1), dtype=torch.float32, device=DEVICE)
            tensors_120x32[region] = t

        return tensors_120x32, cur_tgt


# ==============================================================================
# MAIN LOOP: СИМУЛЯЦИЯ И ВИЗУАЛИЗАЦИЯ
# ==============================================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroCanvas: Heterarchy & Thousand Brains BCI")
    font = pygame.font.SysFont("consolas", 18, bold=True)
    font_sm = pygame.font.SysFont("consolas", 14)

    heterarchy = PrefrontalHeterarchy().to(DEVICE)
    agent = ClosedLoopAgent()

    # Фаза калибровки
    print("🧠 ЗАПУСК КАЛИБРОВКИ (Обучение HTM)...")
    for step in range(80):
        # Принудительно ставим агенту цели
        tgt_idx = (step // 20) % 4
        agent.tgt_idx = tgt_idx
        tensors, tgt = agent.update_and_generate_tensors(tgt['g'], tgt['s'])
        
        # План Б - следующая цель
        plan_b = agent.targets[(tgt_idx + 1) % 4]
        
        heterarchy.learn_calibration(tensors, tgt['g'], tgt['s'], plan_b['g'], plan_b['s'])
        
        screen.fill((15, 20, 30))
        txt = font.render(f"КАЛИБРОВКА HTM SPATIAL POOLER... [{step}/80] Цель: {tgt['name']}", True, (0, 255, 200))
        screen.blit(txt, (50, HEIGHT//2))
        pygame.display.flip()
        time.sleep(0.05)

    print("✅ Калибровка завершена! Переход в режим Closed-Loop.")

    screen_g, screen_s = 0.5, 0.5
    clock = pygame.time.Clock()

    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return pygame.quit()

        # 1. Агент генерирует нейроактивность (tensors) на основе своих желаний
        tensors, target = agent.update_and_generate_tensors(screen_g, screen_s)

        # 2. Мозг (Гетерархия) декодирует тензоры
        pred_g, pred_s, is_plan_b = heterarchy.predict(tensors)

        # 3. Диффузионный "Холст" (Экран) плавно следует за декодированной мыслью
        screen_g = screen_g * 0.85 + pred_g * 0.15
        screen_s = screen_s * 0.85 + pred_s * 0.15

        # --- РЕНДЕРИНГ ---
        screen.fill((15, 20, 30))

        # Холст Латентного Пространства (SD-LCM Proxy)
        canvas_rect = pygame.Rect(400, 150, 500, 500)
        pygame.draw.rect(screen, (30, 40, 50), canvas_rect)
        pygame.draw.rect(screen, (100, 150, 200), canvas_rect, 2)

        # Углы-ориентиры
        screen.blit(font_sm.render("ГОРА", True, (255,255,255)), (360, 130))
        screen.blit(font_sm.render("ЗАМОК", True, (255,255,255)), (910, 130))
        screen.blit(font_sm.render("ОКЕАН", True, (255,255,255)), (360, 660))
        screen.blit(font_sm.render("НЕБОСКРЕБ", True, (255,255,255)), (910, 660))

        # Точка "Экрана" (То, что видит человек)
        px = 400 + int(screen_g * 500)
        py = 150 + int(screen_s * 500)
        pygame.draw.circle(screen, (0, 255, 255), (px, py), 12)
        
        # Точка "Цели Агента" (Куда он хочет смотреть)
        tx = 400 + int(target['g'] * 500)
        ty = 150 + int(target['s'] * 500)
        pygame.draw.circle(screen, (255, 50, 50), (tx, ty), 8, 2)

        # --- ТЕЛЕМЕТРИЯ АГЕНТА И МОЗГА ---
        pygame.draw.rect(screen, (20, 25, 35), (30, 30, 340, 400), border_radius=8)
        pygame.draw.rect(screen, (255, 150, 50), (30, 30, 340, 400), 1, border_radius=8)

        y = 40
        screen.blit(font.render("КОГНИТИВНЫЙ АГЕНТ", True, (255, 150, 50)), (45, y)); y+=30
        screen.blit(font_sm.render(f"Мысль (Цель) : {target['name']}", True, (200, 220, 255)), (45, y)); y+=20
        screen.blit(font_sm.render(f"Состояние    : {agent.mode}", True, (255, 255, 100)), (45, y)); y+=30

        # Полоски состояний
        pygame.draw.rect(screen, (50, 50, 50), (45, y, 200, 10))
        pygame.draw.rect(screen, (100, 255, 100), (45, y, int(agent.satisfaction*200), 10))
        screen.blit(font_sm.render("Satisfaction", True, (150,150,150)), (255, y-3)); y+=25

        pygame.draw.rect(screen, (50, 50, 50), (45, y, 200, 10))
        pygame.draw.rect(screen, (255, 200, 50), (45, y, int(agent.boredom*200), 10))
        screen.blit(font_sm.render("Boredom", True, (150,150,150)), (255, y-3)); y+=25

        pygame.draw.rect(screen, (50, 50, 50), (45, y, 200, 10))
        pygame.draw.rect(screen, (255, 80, 80), (45, y, int(agent.frustration*200), 10))
        screen.blit(font_sm.render("Frustration", True, (150,150,150)), (255, y-3)); y+=40

        # Бенчмарк
        screen.blit(font.render("HTM HETERARCHY", True, (0, 255, 200)), (45, y)); y+=30
        plan_color = (255, 80, 80) if is_plan_b else (100, 255, 100)
        plan_text = "ПЛАН Б (Phase Reset!)" if is_plan_b else "ПЛАН А (Консенсус)"
        screen.blit(font_sm.render(f"Fpz Branch : {plan_text}", True, plan_color), (45, y)); y+=20
        
        # Расчет Retention
        err = math.hypot(pred_g - target['g'], pred_s - target['s'])
        acc = max(0.0, 1.0 - err) * 100.0
        screen.blit(font_sm.render(f"TBT Retention: {acc:.1f}%", True, (0, 255, 255)), (45, y)); y+=20

        pygame.display.flip()

if __name__ == '__main__':
    main()
