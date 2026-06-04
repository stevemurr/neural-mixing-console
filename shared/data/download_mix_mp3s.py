#!/usr/bin/env python3
"""Download Cambridge-MT preview MP3s using a Cloudflare cf_clearance cookie.

Run this on the machine where you obtained the cookie — the cf_clearance
token is bound to that browser's User-Agent (and often IP), so it won't
validate from a different host.

Setup (one-time):
    pip install curl_cffi

Get the cookie from your browser:
    1. Visit https://previews.cambridge-mt.com  (passes the Cloudflare
       challenge once — needs to be the previews subdomain, since
       cf_clearance is scoped per-hostname)
    2. DevTools → Application → Cookies → previews.cambridge-mt.com →
       copy the `cf_clearance` value

Usage:
    export CF_CLEARANCE='paste-cookie-value-here'
    python3 download_mix_mp3s.py --out previews/

The full URL list (605 verified previews) is embedded below — no extra
files needed. Override with --urls FILE or pass extra URLs as positional
args.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

PREVIEW_HOST = "https://previews.cambridge-mt.com"

# User-Agent string that the embedded cookie was issued for. cf_clearance
# is bound to the UA — this MUST match the browser that grabbed the cookie.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/147.0.0.0 Safari/537.36"
)

# 605 verified Cambridge-MT preview filenames (union of harvested page
# index + locally-matched session folders). Joined to PREVIEW_HOST at runtime.
PREVIEW_BASENAMES: list[str] = [
    "1125_Full_Preview.mp3",
    "1OfTheFirst_Full_Preview.mp3",
    "3DMARCoACapella_Full_Preview.mp3",
    "3DMARCoPianoSolo1_Full_Preview.mp3",
    "3DMARCoStringQuartet_Full_Preview.mp3",
    "4OutOf10_Full_Preview.mp3",
    "57Chevy_Full_Preview.mp3",
    "5thFloor_Full_Preview.mp3",
    "64Bristol_Full_Preview.mp3",
    "80sRocker_Full_Preview.mp3",
    "ACryForNormalcy_Full_Preview.mp3",
    "ALittleLate_Full_Preview.mp3",
    "AMiLado_Full_Preview.mp3",
    "APlaceForUs_Full_Preview.mp3",
    "AReasonToLeave_Full_Preview.mp3",
    "ASailorOnceMore_Full_Preview.mp3",
    "AccessDenied_Full_Preview.mp3",
    "Acomatay_Full_Preview.mp3",
    "AeMereHumsafar_Full_Preview.mp3",
    "AeternumVale_Full_Preview.mp3",
    "Africa_Full_Preview.mp3",
    "Afterglow_Full_Preview.mp3",
    "AiguilleRouge_Full_Preview.mp3",
    "Air_Full_Preview.mp3",
    "AlexTheAdventurer_Full_Preview.mp3",
    "AliveYetFree_Full_Preview.mp3",
    "AllAmericanMutt_Full_Preview.mp3",
    "AllIKnow_Full_Preview.mp3",
    "AllINeverWas_Full_Preview.mp3",
    "AllSoulsMoon_Full_Preview.mp3",
    "AllTheGinIsGone_Full_Preview.mp3",
    "AloneWithYou_Full_Preview.mp3",
    "Alright_Full_Preview.mp3",
    "Amalgamize_Full_Preview.mp3",
    "Ambitions_Full_Preview.mp3",
    "Amen_Full_Preview.mp3",
    "AmorQueLucha_Full_Preview.mp3",
    "AnNascNua_Full_Preview.mp3",
    "AnUltraVioletApology_Full_Preview.mp3",
    "AncoraQui_Full_Preview.mp3",
    "Angelsaint_Full_Preview.mp3",
    "Animal_Full_Preview.mp3",
    "Animals_Full_Preview.mp3",
    "AnomalousWeeping_Full_Preview.mp3",
    "AnotherDayCalling_Full_Preview.mp3",
    "AnotherFeeling_Full_Preview.mp3",
    "AnotherLife2_Full_Preview.mp3",
    "AnotherLife_Full_Preview.mp3",
    "AnotherWay_Full_Preview.mp3",
    "AprilBlues_Full_Preview.mp3",
    "Arto_Full_Preview.mp3",
    "Asbestos_Full_Preview.mp3",
    "Asiza_Full_Preview.mp3",
    "Atrophy_Full_Preview.mp3",
    "Attention_Full_Preview.mp3",
    "AureusNecrosis_Full_Preview.mp3",
    "Away_Full_Preview.mp3",
    "AyniNehirde_Full_Preview.mp3",
    "BackDown_Full_Preview.mp3",
    "BackFromTheStart_Full_Preview.mp3",
    "BackHomeToBlue_Full_Preview.mp3",
    "BackInTime_Full_Preview.mp3",
    "BackToTheNineties_Full_Preview.mp3",
    "BackroomInTulsa_Full_Preview.mp3",
    "BananaSplit_Full_Preview.mp3",
    "Bankroll_Full_Preview.mp3",
    "Bannockburn_Full_Preview.mp3",
    "Beauty_Full_Preview.mp3",
    "Believe_Full_Preview.mp3",
    "Believer_Full_Preview.mp3",
    "Bess_Full_Preview.mp3",
    "BetterWay_Full_Preview.mp3",
    "Better_Full_Preview.mp3",
    "BigBang_Full_Preview.mp3",
    "BigMansDeath_Full_Preview.mp3",
    "BitchIsParanoid_Full_Preview.mp3",
    "Bitter_Full_Preview.mp3",
    "BlackOutBetty_Full_Preview.mp3",
    "BloodToBone_Full_Preview.mp3",
    "Bloodshed_Full_Preview.mp3",
    "Blue_Full_Preview.mp3",
    "BoatRide_Full_Preview.mp3",
    "Borderline_Full_Preview.mp3",
    "BrightAngel_Full_Preview.mp3",
    "BrokenAgain_Full_Preview.mp3",
    "BrokenMan_Full_Preview.mp3",
    "BuildinItUp_Full_Preview.mp3",
    "BurialOfSilence_Full_Preview.mp3",
    "BurningBridges_Full_Preview.mp3",
    "ByMySide_Full_Preview.mp3",
    "CUNextTime_Full_Preview.mp3",
    "Cage_Full_Preview.mp3",
    "CanYouSayTheSame_Full_Preview.mp3",
    "Canon_Full_Preview.mp3",
    "CarnivalCharade_Full_Preview.mp3",
    "CarolinaInThePines_Full_Preview.mp3",
    "CarouselRide_Full_Preview.mp3",
    "Cascade_Full_Preview.mp3",
    "CatchTheWave_Full_Preview.mp3",
    "CatchingUp_Full_Preview.mp3",
    "CelloConcerto_Full_Preview.mp3",
    "CentauriB_Full_Preview.mp3",
    "Change_Full_Preview.mp3",
    "ChangingThings_Full_Preview.mp3",
    "ChardonnayLektrixRemix_Full_Preview.mp3",
    "ChardonnayMystikVibeRemix_Full_Preview.mp3",
    "ChardonnayOrneryRemix_Full_Preview.mp3",
    "Chardonnay_Full_Preview.mp3",
    "Chasque_Full_Preview.mp3",
    "ChristmasBlues_Full_Preview.mp3",
    "Cinderella_Full_Preview.mp3",
    "ClinicA_Full_Preview.mp3",
    "CogInTheMachine_Full_Preview.mp3",
    "ColdStrive_Full_Preview.mp3",
    "Cold_Full_Preview.mp3",
    "CollideMoniBlue_Full_Preview.mp3",
    "Collide_Full_Preview.mp3",
    "ColourMeRed_Full_Preview.mp3",
    "Comet_Full_Preview.mp3",
    "ComfortLivesInBelief_Full_Preview.mp3",
    "Contact_Full_Preview.mp3",
    "Contraband_Full_Preview.mp3",
    "Convertible_Full_Preview.mp3",
    "Copper_Full_Preview.mp3",
    "Core_Full_Preview.mp3",
    "CorineCorine_Full_Preview.mp3",
    "CorsDesbocats_Full_Preview.mp3",
    "CrazyForMe_Full_Preview.mp3",
    "CrazyGirl_Full_Preview.mp3",
    "Crazy_Full_Preview.mp3",
    "CrucialTaunt_Full_Preview.mp3",
    "CruisingTheIce_Full_Preview.mp3",
    "CryingRivers_Full_Preview.mp3",
    "CurAnLongAgSeol_Full_Preview.mp3",
    "CyberMower_Full_Preview.mp3",
    "Cybermod_Full_Preview.mp3",
    "DaddyD_Full_Preview.mp3",
    "DaisyDaisy_Full_Preview.mp3",
    "DarkHorses_Full_Preview.mp3",
    "DarkIllusion_Full_Preview.mp3",
    "DarkSpace_Full_Preview.mp3",
    "DasFunkeren_Full_Preview.mp3",
    "DeadEnemies_Full_Preview.mp3",
    "DeadMensRide_Full_Preview.mp3",
    "DeadRoses_Full_Preview.mp3",
    "DeathMetalSucks_Full_Preview.mp3",
    "DenyControl_Full_Preview.mp3",
    "Despierta_Full_Preview.mp3",
    "DevilsWords_Full_Preview.mp3",
    "DieYoung_Full_Preview.mp3",
    "DiosSalveElRockAndRollCieloSinSantos_Full_Preview.mp3",
    "DirectExperience_Full_Preview.mp3",
    "Disconnected_Full_Preview.mp3",
    "DivesAndLazarus_Full_Preview.mp3",
    "DoItAgain_Full_Preview.mp3",
    "DoNotStand_Full_Preview.mp3",
    "DondeNoLlegaLaLuz_Full_Preview.mp3",
    "DontCare_Full_Preview.mp3",
    "DontLetGo_Full_Preview.mp3",
    "DontLetTheDevilTakeYourMind_Full_Preview.mp3",
    "DontPleaseStay_Full_Preview.mp3",
    "DontPutMeOnHold_Full_Preview.mp3",
    "Dorothy_Full_Preview.mp3",
    "Downtempo_Full_Preview.mp3",
    "DragMeDown_Full_Preview.mp3",
    "Drag_Full_Preview.mp3",
    "DreamNo12_Full_Preview.mp3",
    "DreamState_Full_Preview.mp3",
    "Dreamland_Full_Preview.mp3",
    "Droid_Full_Preview.mp3",
    "DumberByTheMinute_Full_Preview.mp3",
    "DuneRider_Full_Preview.mp3",
    "DustYouAndMe_Full_Preview.mp3",
    "DutyAndMemories_Full_Preview.mp3",
    "DyingLight_Full_Preview.mp3",
    "EasyTiger_Full_Preview.mp3",
    "Echo_Full_Preview.mp3",
    "Echoes_Full_Preview.mp3",
    "Ecstasy_Full_Preview.mp3",
    "ElMarinero_Full_Preview.mp3",
    "Electrvm_Full_Preview.mp3",
    "ElizaJane_Full_Preview.mp3",
    "EnDance_Full_Preview.mp3",
    "Encore_Full_Preview.mp3",
    "Entwine_Full_Preview.mp3",
    "Equilibrium_Full_Preview.mp3",
    "Error404_Full_Preview.mp3",
    "EscapeOfTheHoppingRhinos_Full_Preview.mp3",
    "EstoSeLlamaVivir_Full_Preview.mp3",
    "Etc_Full_Preview.mp3",
    "Eventide_Full_Preview.mp3",
    "EvilBones_Full_Preview.mp3",
    "Excessive_Full_Preview.mp3",
    "Expired_Full_Preview.mp3",
    "Eyeliner_Full_Preview.mp3",
    "Eyes_Full_Preview.mp3",
    "FacingTheTruth_Full_Preview.mp3",
    "Fade_Full_Preview.mp3",
    "Fairytale_Full_Preview.mp3",
    "Femme_Full_Preview.mp3",
    "Fever_Full_Preview.mp3",
    "FishTacos_Full_Preview.mp3",
    "Flames_Full_Preview.mp3",
    "FlawedDesign_Full_Preview.mp3",
    "Flawed_Full_Preview.mp3",
    "FlecheDOr_Full_Preview.mp3",
    "FleshAndBone_Full_Preview.mp3",
    "FloresDeAbril_Full_Preview.mp3",
    "FlyHigh_Full_Preview.mp3",
    "FollowMe_Full_Preview.mp3",
    "Fool_Full_Preview.mp3",
    "ForIAmTheMoon_Full_Preview.mp3",
    "FountainOfEternalLife_Full_Preview.mp3",
    "FourGraham_Full_Preview.mp3",
    "FragileThoughts_Full_Preview.mp3",
    "Fragments_Full_Preview.mp3",
    "Freedom_Full_Preview.mp3",
    "FromEarthToPlanetOmega_Full_Preview.mp3",
    "Funkkihillo_Full_Preview.mp3",
    "FunkyToTheBone_Full_Preview.mp3",
    "FutureSoil_Full_Preview.mp3",
    "GOTGF_Full_Preview.mp3",
    "GetFooled_Full_Preview.mp3",
    "GetOutOfBed_Full_Preview.mp3",
    "GhostBitch_Full_Preview.mp3",
    "Gimme_Full_Preview.mp3",
    "GoGoGo_Full_Preview.mp3",
    "GoingToTheMoon_Full_Preview.mp3",
    "Gokyuzu_Full_Preview.mp3",
    "Gone_Full_Preview.mp3",
    "GoodTime_Full_Preview.mp3",
    "GoregasmicGrotesqueries_Full_Preview.mp3",
    "GotYourLove_Full_Preview.mp3",
    "Gravediggers_Full_Preview.mp3",
    "GuerraAllaFrontiera_Full_Preview.mp3",
    "HammerDown_Full_Preview.mp3",
    "HappyPills_Full_Preview.mp3",
    "HateSongs_Full_Preview.mp3",
    "HauntedAge_Full_Preview.mp3",
    "HauntedHouse_Full_Preview.mp3",
    "Hawaii_Full_Preview.mp3",
    "Headspace_Full_Preview.mp3",
    "HeartOfMyHomeTown_Full_Preview.mp3",
    "HeartOnMyThumb_Full_Preview.mp3",
    "HeartPeripheral_Full_Preview.mp3",
    "Heartbeats_Full_Preview.mp3",
    "HeatherJane_Full_Preview.mp3",
    "HeyCarrieAnne_Full_Preview.mp3",
    "HeyDelilah_Full_Preview.mp3",
    "HoldMe2_Full_Preview.mp3",
    "HoldMe_Full_Preview.mp3",
    "HoldOnYou_Full_Preview.mp3",
    "HomeInTheCountry_Full_Preview.mp3",
    "Homebound_Full_Preview.mp3",
    "HopeAndTheSea_Full_Preview.mp3",
    "Horizon_Full_Preview.mp3",
    "HowToMakeAMirror_Full_Preview.mp3",
    "HumanMistakes_Full_Preview.mp3",
    "HungarianDanceNo5_Full_Preview.mp3",
    "Hurricane_Full_Preview.mp3",
    "IAmTheDesert_Full_Preview.mp3",
    "IdRatherBeDrinkin_Full_Preview.mp3",
    "IfIWereABell_Full_Preview.mp3",
    "IfYouSay_Full_Preview.mp3",
    "IfYouThenI_Full_Preview.mp3",
    "IllFate_Full_Preview.mp3",
    "ImAlright_Full_Preview.mp3",
    "ImComingHome_Full_Preview.mp3",
    "InTheBand_Full_Preview.mp3",
    "IncidenteEnIntag_Full_Preview.mp3",
    "Ingloria_Full_Preview.mp3",
    "InnerCircle_Full_Preview.mp3",
    "Interlude_Full_Preview.mp3",
    "IntoMyDreams_Full_Preview.mp3",
    "IntoTheForest_Full_Preview.mp3",
    "IsYouIsOrIsYouAint_Full_Preview.mp3",
    "Islets_Full_Preview.mp3",
    "ItWasMyFaultForWaiting_Full_Preview.mp3",
    "ItsInTheseTimes_Full_Preview.mp3",
    "ItsMyRight_Full_Preview.mp3",
    "ItsSoEasyToLoveYou_Full_Preview.mp3",
    "JaMakeYaDance_Full_Preview.mp3",
    "JapanSong_Full_Preview.mp3",
    "JedenWinter_Full_Preview.mp3",
    "JesuJoy_Full_Preview.mp3",
    "JoesBar_Full_Preview.mp3",
    "JohnDoesBlues_Full_Preview.mp3",
    "JoyRide_Full_Preview.mp3",
    "JumpAcross_Full_Preview.mp3",
    "JustDontTalk_Full_Preview.mp3",
    "JustLetItGo_Full_Preview.mp3",
    "JustOneMinute_Full_Preview.mp3",
    "KakTvoiDelaVova_Full_Preview.mp3",
    "KaneGuru_Full_Preview.mp3",
    "KeepsFlowing_Full_Preview.mp3",
    "KingOfTheWeekend_Full_Preview.mp3",
    "KingRascal_Full_Preview.mp3",
    "KingsAndQueens_Full_Preview.mp3",
    "Knockout_Full_Preview.mp3",
    "Koishii_Full_Preview.mp3",
    "Lacuna_Full_Preview.mp3",
    "LastNightsGig_Full_Preview.mp3",
    "LearningHowToFly_Full_Preview.mp3",
    "LeftBlind_Full_Preview.mp3",
    "LetTheMusicFadeAway_Full_Preview.mp3",
    "LetsDance_Full_Preview.mp3",
    "LieToMe_Full_Preview.mp3",
    "LifeGetsInTheWay_Full_Preview.mp3",
    "LightsOut_Full_Preview.mp3",
    "LikeYouDo_Full_Preview.mp3",
    "LittleLighter_Full_Preview.mp3",
    "LittleWing_Full_Preview.mp3",
    "LivingInTheCity_Full_Preview.mp3",
    "LivingLie_Full_Preview.mp3",
    "LocationLocation_Full_Preview.mp3",
    "LongOverdue_Full_Preview.mp3",
    "LongRoad_Full_Preview.mp3",
    "LongWayHome_Full_Preview.mp3",
    "LookinToughFeelinGood_Full_Preview.mp3",
    "LookoutMountain_Full_Preview.mp3",
    "LostMyWay_Full_Preview.mp3",
    "LuaNegra_Full_Preview.mp3",
    "MacacoRegresso_Full_Preview.mp3",
    "MachinesOnTreadmills_Full_Preview.mp3",
    "Magdalena_Full_Preview.mp3",
    "MaggieMay_Full_Preview.mp3",
    "Magilla_Full_Preview.mp3",
    "MakinWhoopee_Full_Preview.mp3",
    "MarshMarigoldsSong_Full_Preview.mp3",
    "Mathematician_Full_Preview.mp3",
    "Matterplay_Full_Preview.mp3",
    "Maya_Full_Preview.mp3",
    "Maybe_Full_Preview.mp3",
    "MeAndMyCrew_Full_Preview.mp3",
    "Melancholy_Full_Preview.mp3",
    "MeuBem_Full_Preview.mp3",
    "MikesSulking_Full_Preview.mp3",
    "MilkCowBlues_Full_Preview.mp3",
    "MisterMister_Full_Preview.mp3",
    "MorbusAnimi_Full_Preview.mp3",
    "MorningSickness_Full_Preview.mp3",
    "MuchTooMuch_Full_Preview.mp3",
    "MuddyWater_Full_Preview.mp3",
    "MustBeVoodoo_Full_Preview.mp3",
    "Mute_Full_Preview.mp3",
    "MyFatherNeverLovedMe_Full_Preview.mp3",
    "MyOwn_Full_Preview.mp3",
    "Nalim_Full_Preview.mp3",
    "Naturally_Full_Preview.mp3",
    "NearlyThere_Full_Preview.mp3",
    "NeverEbbButflow_Full_Preview.mp3",
    "NeverLeaveTheNightAlone_Full_Preview.mp3",
    "NeverLetYouGo_Full_Preview.mp3",
    "NeverStop_Full_Preview.mp3",
    "NewDayDawning_Full_Preview.mp3",
    "NoGrip_Full_Preview.mp3",
    "NoLimits_Full_Preview.mp3",
    "NonLoDiroColLabbro_Full_Preview.mp3",
    "NosPalpitants_Full_Preview.mp3",
    "NossoMundoDeixouDeExistir_Full_Preview.mp3",
    "Nostalgic_Full_Preview.mp3",
    "NotAlone_Full_Preview.mp3",
    "NothingAtAll_Full_Preview.mp3",
    "Nowhere_Full_Preview.mp3",
    "Nuvole_Full_Preview.mp3",
    "OdiALaBarretina_Full_Preview.mp3",
    "OfIceAndHopelessFate_Full_Preview.mp3",
    "OhLife_Full_Preview.mp3",
    "Oil_Full_Preview.mp3",
    "OldEmptyNest_Full_Preview.mp3",
    "Omen_Full_Preview.mp3",
    "OnceMore_Full_Preview.mp3",
    "OneFlipFlop_Full_Preview.mp3",
    "OneMinuteSmile_Full_Preview.mp3",
    "OneOfTheseDays_Full_Preview.mp3",
    "OneTimeWeekend_Full_Preview.mp3",
    "OnesAndZeroes_Full_Preview.mp3",
    "OpenFire_Full_Preview.mp3",
    "OrsonWelles_Full_Preview.mp3",
    "OurLoveIsHereToStay_Full_Preview.mp3",
    "OutaControl_Full_Preview.mp3",
    "Outer_Full_Preview.mp3",
    "OverTheTop_Full_Preview.mp3",
    "OwnWayToBoogie_Full_Preview.mp3",
    "PaddyFahysJig_Full_Preview.mp3",
    "PainRemains_Full_Preview.mp3",
    "Paraisso_Full_Preview.mp3",
    "Paris_Full_Preview.mp3",
    "ParoleVuote_Full_Preview.mp3",
    "PassengerSide_Full_Preview.mp3",
    "PassingShips_Full_Preview.mp3",
    "Pennies_Full_Preview.mp3",
    "PerPoderTeCantar_Full_Preview.mp3",
    "PetitAbisme_Full_Preview.mp3",
    "Phantom2_Full_Preview.mp3",
    "PhoneRage_Full_Preview.mp3",
    "PiaMater_Full_Preview.mp3",
    "PianoConcertoK414_Full_Preview.mp3",
    "PieceOfMe_Full_Preview.mp3",
    "Piers_Full_Preview.mp3",
    "Pirarucumbia_Full_Preview.mp3",
    "Place2Be_Full_Preview.mp3",
    "PlacidoDomingo_Full_Preview.mp3",
    "PlaeseHaalp_Full_Preview.mp3",
    "Plums_Full_Preview.mp3",
    "Polemic_Full_Preview.mp3",
    "Pony_Full_Preview.mp3",
    "PoorBoy_Full_Preview.mp3",
    "PostRockIsDumb_Full_Preview.mp3",
    "PrayForTheRain_Full_Preview.mp3",
    "PreachRightHere_Full_Preview.mp3",
    "Prisoner_Full_Preview.mp3",
    "Prodigal_Full_Preview.mp3",
    "Progresivo1ElVuelo_Full_Preview.mp3",
    "Purgatory_Full_Preview.mp3",
    "PushAndPull_Full_Preview.mp3",
    "Puzzle_Full_Preview.mp3",
    "QueensLight_Full_Preview.mp3",
    "Quicksand_Full_Preview.mp3",
    "Rachel_Full_Preview.mp3",
    "RainyDayII_Full_Preview.mp3",
    "RaspberryJam_Full_Preview.mp3",
    "RatRace_Full_Preview.mp3",
    "RedOnYou_Full_Preview.mp3",
    "Reflection_Full_Preview.mp3",
    "Release_Full_Preview.mp3",
    "Relentlessly_Full_Preview.mp3",
    "RescueMe_Full_Preview.mp3",
    "ResistenciaParaUnNuevoComenzar_Full_Preview.mp3",
    "RestInHell_Full_Preview.mp3",
    "ResurrectionResurrected_Full_Preview.mp3",
    "Retry_Full_Preview.mp3",
    "Revelations_Full_Preview.mp3",
    "RevoX_Full_Preview.mp3",
    "Riesling_Full_Preview.mp3",
    "RiverOfTheWhiteGloom_Full_Preview.mp3",
    "RiversRisin_Full_Preview.mp3",
    "Roar_Full_Preview.mp3",
    "Rockshow_Full_Preview.mp3",
    "Roma_Full_Preview.mp3",
    "RootsOfMankind_Full_Preview.mp3",
    "RumbaChonta_Full_Preview.mp3",
    "RunningOut_Full_Preview.mp3",
    "RussianBot_Full_Preview.mp3",
    "SameKindOfLife_Full_Preview.mp3",
    "SandcastlesIllusion_Full_Preview.mp3",
    "Sandstorm_Full_Preview.mp3",
    "SantJordi_Full_Preview.mp3",
    "SantaFe_Full_Preview.mp3",
    "Santa_Full_Preview.mp3",
    "Sascha_Full_Preview.mp3",
    "SaudadeDoTeuBeijo_Full_Preview.mp3",
    "Scar_Full_Preview.mp3",
    "Scarlett_Full_Preview.mp3",
    "SeatBack_Full_Preview.mp3",
    "Semantics_Full_Preview.mp3",
    "SeptemberTrance_Full_Preview.mp3",
    "SetMeFree_Full_Preview.mp3",
    "SevenFeel_Full_Preview.mp3",
    "ShelaYaMarhaba_Full_Preview.mp3",
    "Shore_Full_Preview.mp3",
    "Showmonster_Full_Preview.mp3",
    "ShutUpAndPlay_Full_Preview.mp3",
    "SinSix_Full_Preview.mp3",
    "Siren_Full_Preview.mp3",
    "Sirens_Full_Preview.mp3",
    "Sirenz_Full_Preview.mp3",
    "Slapback_Full_Preview.mp3",
    "SleepByTheFireBloomInWater_Full_Preview.mp3",
    "SleighRide_Full_Preview.mp3",
    "SlowDown_Full_Preview.mp3",
    "SmokerAndTheStoner_Full_Preview.mp3",
    "SoHiSoLo_Full_Preview.mp3",
    "SodaEnvy_Full_Preview.mp3",
    "SomeDay_Full_Preview.mp3",
    "SomeTrashyThrashIGuess_Full_Preview.mp3",
    "SongForJohn_Full_Preview.mp3",
    "SongOfIndia_Full_Preview.mp3",
    "SorryGirl_Full_Preview.mp3",
    "Sorry_Full_Preview.mp3",
    "SouthOfTheWater_Full_Preview.mp3",
    "Spaces_Full_Preview.mp3",
    "SpiritCold_Full_Preview.mp3",
    "Stalker_Full_Preview.mp3",
    "SteampunkSiege_Full_Preview.mp3",
    "StillFlyin_Full_Preview.mp3",
    "StopAndRise_Full_Preview.mp3",
    "StruggleCity_Full_Preview.mp3",
    "SuchFinePeople_Full_Preview.mp3",
    "SugarFaith_Full_Preview.mp3",
    "Sugar_Full_Preview.mp3",
    "SuitYou_Full_Preview.mp3",
    "Summerghost_Full_Preview.mp3",
    "Summertime2_Full_Preview.mp3",
    "Summertime_Full_Preview.mp3",
    "SunDrenched_Full_Preview.mp3",
    "Sunshine_Full_Preview.mp3",
    "Surrendering_Full_Preview.mp3",
    "SymphonyOfSilence_Full_Preview.mp3",
    "SzaradASzaj_Full_Preview.mp3",
    "TakeItOff_Full_Preview.mp3",
    "TearsInTheRain_Full_Preview.mp3",
    "Technomantra_Full_Preview.mp3",
    "TelemannLaLyraOverture_Full_Preview.mp3",
    "Teleport_Full_Preview.mp3",
    "TellMeNice_Full_Preview.mp3",
    "TemporaryHappiness_Full_Preview.mp3",
    "TeraniaCreekWalking_Full_Preview.mp3",
    "Tgheer_Full_Preview.mp3",
    "ThatsEntertainment_Full_Preview.mp3",
    "ThatsHowIGotToMemphis_Full_Preview.mp3",
    "TheBluesIsALady_Full_Preview.mp3",
    "TheCalling_Full_Preview.mp3",
    "TheCalm_Full_Preview.mp3",
    "TheCrossRoads_Full_Preview.mp3",
    "TheCrown_Full_Preview.mp3",
    "TheDarkAbyss_Full_Preview.mp3",
    "TheDeathlessOne_Full_Preview.mp3",
    "TheDice_Full_Preview.mp3",
    "TheElephant_Full_Preview.mp3",
    "TheFeeling_Full_Preview.mp3",
    "TheForthcomingTurn_Full_Preview.mp3",
    "TheGlass_Full_Preview.mp3",
    "TheHardestPart_Full_Preview.mp3",
    "TheIslandOfAlsocanla_Full_Preview.mp3",
    "TheLastElm_Full_Preview.mp3",
    "TheLastStand_Full_Preview.mp3",
    "TheMachiavellian_Full_Preview.mp3",
    "TheNarcissist_Full_Preview.mp3",
    "TheOpener_Full_Preview.mp3",
    "ThePsychopath_Full_Preview.mp3",
    "TheSagaOfHarrisonCrabfeathers_Full_Preview.mp3",
    "TheTruth_Full_Preview.mp3",
    "TheWeight_Full_Preview.mp3",
    "TheWell_Full_Preview.mp3",
    "TheWind_Full_Preview.mp3",
    "ThisTown_Full_Preview.mp3",
    "ThursdayReverie_Full_Preview.mp3",
    "TimelessPart1_Full_Preview.mp3",
    "TimelessPart2_Full_Preview.mp3",
    "TipToeThroughTheCrypto_Full_Preview.mp3",
    "Tiring_Full_Preview.mp3",
    "ToSamRawfers_Full_Preview.mp3",
    "ToTheWolves_Full_Preview.mp3",
    "Tochka_Full_Preview.mp3",
    "TodaysTheDay_Full_Preview.mp3",
    "TogetherAlone_Full_Preview.mp3",
    "TooBright_Full_Preview.mp3",
    "TooMuch2_Full_Preview.mp3",
    "Tourism_Full_Preview.mp3",
    "Toxic_Full_Preview.mp3",
    "Transcention_Full_Preview.mp3",
    "Trapped_Full_Preview.mp3",
    "Treadmills_Full_Preview.mp3",
    "Triba-Bao-Kuku_Full_Preview.mp3",
    "TrudeTheBumblebee_Full_Preview.mp3",
    "Truth_Full_Preview.mp3",
    "TubosDeCristal_Full_Preview.mp3",
    "TurnOnMe_Full_Preview.mp3",
    "TwoBareHands_Full_Preview.mp3",
    "Ubiquitous_Full_Preview.mp3",
    "Ukraina_Full_Preview.mp3",
    "UnaSemanaSinTi_Full_Preview.mp3",
    "UnaVitaSola_Full_Preview.mp3",
    "Unbroken_Full_Preview.mp3",
    "UnfinishedDreams_Full_Preview.mp3",
    "Unseen_Full_Preview.mp3",
    "UntilIGetBack_Full_Preview.mp3",
    "UpperHand_Full_Preview.mp3",
    "UpsideDown_Full_Preview.mp3",
    "VaLaisseCoulerMesLarmes_Full_Preview.mp3",
    "VirusEnMi_Full_Preview.mp3",
    "ViviendoDelReves_Full_Preview.mp3",
    "VoiCheSapete_Full_Preview.mp3",
    "VoicelessSiren_Full_Preview.mp3",
    "Wahnmal_Full_Preview.mp3",
    "WalkieTalkie_Full_Preview.mp3",
    "WallpaperBaby_Full_Preview.mp3",
    "Waves_Full_Preview.mp3",
    "WayOfLife_Full_Preview.mp3",
    "WayfaringStranger_Full_Preview.mp3",
    "WealthyInTime_Full_Preview.mp3",
    "WellTalkAboutItAllTonight_Full_Preview.mp3",
    "WhatChildIsThis_Full_Preview.mp3",
    "WhatIWant_Full_Preview.mp3",
    "Whiptails_Full_Preview.mp3",
    "WhisperToAScream_Full_Preview.mp3",
    "WhoIAm_Full_Preview.mp3",
    "WhosWhoInHell_Full_Preview.mp3",
    "WhyDontYouStay_Full_Preview.mp3",
    "Wicked_Full_Preview.mp3",
    "Wickerman_Full_Preview.mp3",
    "Widow_Full_Preview.mp3",
    "Window2_Full_Preview.mp3",
    "Window_Full_Preview.mp3",
    "WindsOfGypsyMoor_Full_Preview.mp3",
    "WoodwormSong_Full_Preview.mp3",
    "Wrong_Full_Preview.mp3",
    "XXXV_Full_Preview.mp3",
    "YouAndMeAndTheRadio_Full_Preview.mp3",
    "YouAreTheOne_Full_Preview.mp3",
    "YouDontKnow_Full_Preview.mp3",
    "YouKnowBetter_Full_Preview.mp3",
    "YouMakeMeSmile_Full_Preview.mp3",
    "YourStar_Full_Preview.mp3",
]


def embedded_urls() -> list[str]:
    return [f"{PREVIEW_HOST}/{name}" for name in PREVIEW_BASENAMES]


def load_urls(path: Path | None, extra: list[str]) -> list[str]:
    urls: list[str] = []
    if path:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
    else:
        urls.extend(embedded_urls())
    urls.extend(extra)
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def parse_cookie(value: str) -> dict[str, str]:
    """Accept a bare cf_clearance value or a full 'k=v; k=v' Cookie header."""
    if "=" not in value:
        return {"cf_clearance": value}
    cookies: dict[str, str] = {}
    for pair in value.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        cookies[k.strip()] = v.strip()
    return cookies


def fetch_one(url: str, out_dir: Path, *, cookies, ua, timeout, referer):
    from curl_cffi import requests as cr  # type: ignore

    fname = os.path.basename(urlparse(url).path) or "download.mp3"
    dest = out_dir / fname
    if dest.exists() and dest.stat().st_size > 0:
        return ("skip", url, dest, dest.stat().st_size, "exists")

    headers = {
        "User-Agent": ua,
        "Referer": referer,
        "Accept": "audio/mpeg,audio/*;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        r = cr.get(url, headers=headers, cookies=cookies,
                   impersonate="chrome131", timeout=timeout, stream=True)
    except Exception as e:
        return ("fail", url, dest, 0, f"connection: {e}")

    if r.status_code != 200:
        body = r.content[:160] if hasattr(r, "content") else b""
        return ("fail", url, dest, 0, f"HTTP {r.status_code} {body!r:.180}")

    ctype = r.headers.get("content-type", "")
    if "html" in ctype.lower():
        return ("fail", url, dest, 0,
                f"got HTML (challenge / cookie expired?) ct={ctype}")

    n = 0
    try:
        with open(tmp, "wb") as f:
            for buf in r.iter_content(chunk_size=1 << 20):
                if buf:
                    f.write(buf)
                    n += len(buf)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return ("fail", url, dest, 0, f"write: {e}")
    tmp.rename(dest)
    return ("ok", url, dest, n, "ok")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download Cambridge-MT preview MP3s using a cf_clearance cookie.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if __doc__ else None,
    )
    ap.add_argument("--urls", type=Path,
                    help="text file of URLs (one per line, # comments OK). "
                         "Default: 605 embedded preview URLs.")
    ap.add_argument("urls_extra", nargs="*", metavar="URL",
                    help="additional URLs as positional args")
    ap.add_argument("--out", type=Path, default=Path("./mix-mp3s"),
                    help="output directory (default: ./mix-mp3s)")
    ap.add_argument("--cookie", default=os.environ.get("CF_CLEARANCE", ""),
                    help="cf_clearance value or full Cookie header. "
                         "Default: $CF_CLEARANCE")
    ap.add_argument("--user-agent", default=os.environ.get(
                        "CF_USER_AGENT", DEFAULT_USER_AGENT),
                    help="User-Agent that the cookie was issued for. "
                         "Default: $CF_USER_AGENT, falling back to the "
                         "Mac/Chrome 147 string baked into this script.")
    ap.add_argument("--referer", default="https://cambridge-mt.com/")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="parallel download workers (default: 4 — be polite)")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="per-request timeout in seconds (default: 180)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of fetches (0 = no cap; useful to test)")
    args = ap.parse_args()

    try:
        import curl_cffi  # noqa: F401
    except ImportError:
        print("ERROR: curl_cffi not installed.  pip install curl_cffi", file=sys.stderr)
        return 2

    if not args.cookie:
        print("ERROR: need --cookie or $CF_CLEARANCE", file=sys.stderr)
        print("  Get it from DevTools → Application → Cookies → "
              "previews.cambridge-mt.com → cf_clearance", file=sys.stderr)
        return 2

    urls = load_urls(args.urls, args.urls_extra)
    if not urls:
        print("ERROR: no URLs (use --urls FILE or pass URLs as args)", file=sys.stderr)
        return 2
    if args.limit:
        urls = urls[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    cookies = parse_cookie(args.cookie)

    print(f"target dir:  {args.out.resolve()}")
    print(f"urls:        {len(urls)}")
    print(f"concurrency: {args.concurrency}")
    print(f"user-agent:  {args.user_agent}")
    print(f"cookies:     {sorted(cookies.keys())}")
    print()

    ok = fail = skip = 0
    bytes_total = 0
    failures: list[tuple[str, str]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(fetch_one, u, args.out,
                          cookies=cookies, ua=args.user_agent,
                          timeout=args.timeout, referer=args.referer)
                for u in urls]
        for fut in as_completed(futs):
            status, url, dest, nbytes, msg = fut.result()
            if status == "ok":
                ok += 1
                bytes_total += nbytes
                print(f"  OK    {dest.name:60s} {nbytes/1e6:7.2f} MB")
            elif status == "skip":
                skip += 1
                print(f"  skip  {dest.name:60s} exists ({nbytes/1e6:.2f} MB)")
            else:
                fail += 1
                failures.append((url, msg))
                print(f"  FAIL  {url}\n        {msg}", file=sys.stderr)

    dt = time.time() - t0
    print()
    print(f"done in {dt:.1f}s  ok={ok}  skip={skip}  fail={fail}  "
          f"({bytes_total/1e6:.1f} MB)")
    if failures:
        log = args.out / "_failures.txt"
        with open(log, "w") as f:
            for url, msg in failures:
                f.write(f"{url}\t{msg}\n")
        print(f"failure list written to {log}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
