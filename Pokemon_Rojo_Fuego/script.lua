
function pokemon_fire_red_done ()
    if data.EnemyHP1 == 0 then
        if data.EnemyHP2 == 0 then
            if data.EnemyHP3 == 0 then
                if data.EnemyHP4 == 0 then
                    if data.EnemyHP5 == 0 then
                        if data.EnemyHP6 == 0 then
                            return true
                        end
                    end
                end
            end
        end    
    end

    if data.HP1 == 0 then
        if data.HP2 == 0 then
            if data.HP3 == 0 then
                if data.HP4 == 0 then
                    if data.HP5 == 0 then
                        if data.HP6 == 0 then
                            return true
                        end
                    end
                end
            end
        end    
    end
        
    return false

end